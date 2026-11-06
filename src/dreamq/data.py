"""Schema mapping, molecular identity, audit, and label boundaries."""
from __future__ import annotations

import hashlib
import itertools
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from .utils import digest, file_digest, write_json

LOG = logging.getLogger(__name__)
AUDIT_VERSION = 2


def canonicalize(smiles: str) -> dict[str, Any]:
    if not isinstance(smiles, str) or not smiles.strip():
        return {"canonical_smiles": None, "structure_id": None, "scaffold": None, "error": "empty SMILES"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return {"canonical_smiles": None, "structure_id": None, "scaffold": None, "error": "invalid SMILES"}
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # Splitting is conservative: stereoisomers share a non-isomeric scaffold.
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return {"canonical_smiles": canonical,
            "structure_id": hashlib.sha256(canonical.encode()).hexdigest(),
            "scaffold": scaffold or "__ACYCLIC__", "error": None}


def read_table(path: Path, mapping: dict, labeled: bool) -> pd.DataFrame:
    # usecols is an explicit boundary: never read any labels from other tables.
    columns = [mapping["id"], mapping["smiles"]]
    if labeled:
        columns.append(mapping["label"])
    raw = pd.read_csv(path, usecols=columns, dtype={mapping["id"]: str, mapping["smiles"]: str})
    frame = raw.rename(columns={mapping["id"]: "compound_id", mapping["smiles"]: "smiles",
                                **({mapping["label"]: "label"} if labeled else {})})
    frame["smiles"] = frame.smiles.fillna("")
    if frame.compound_id.isna().any() or frame.compound_id.str.strip().eq("").any():
        raise ValueError(f"Missing compound identifiers in {path}")
    if labeled:
        numeric = pd.to_numeric(frame.label, errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError("Labeled activities must be binary 0/1 and present on every row")
        frame["label"] = numeric.astype(np.int8)
    frame["row_index"] = np.arange(len(frame))
    return frame


def _manifest(config: dict) -> list[dict]:
    root = Path(config["data_dir"])
    manifest = pd.read_csv(root / config["data"]["manifest"])
    if "name" not in manifest:
        raise ValueError("Manifest must include the file name column 'name'")
    records = []
    for role in ["labeled", "unlabeled", "leaderboard"]:
        filename = config["data"][role]["file"]
        rows = manifest.loc[manifest.name == filename]
        if len(rows) != 1:
            raise ValueError(f"Manifest must contain exactly one entry for {filename}")
        path = root / filename
        entry = rows.iloc[0].fillna("").to_dict()
        observed = hashlib.md5(path.read_bytes()).hexdigest()
        expected = str(entry.get("dataFileMD5Hex", "")).strip()
        if expected and observed != expected:
            raise ValueError(f"Manifest MD5 mismatch for {filename}")
        entry.update(role=role, observed_md5=observed, sha256=file_digest(path))
        records.append(entry)
    return records


def audit(config: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    root, cache = Path(config["data_dir"]), Path(config["cache_dir"])
    report_dir = Path(config["artifact_dir"]) / "audit"
    manifest = _manifest(config)
    signature = digest({"files": manifest, "mapping": config["data"], "version": AUDIT_VERSION,
                        "rdkit": rdBase.rdkitVersion})
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / f"audit-{signature}.joblib"
    if cache_path.exists():
        tables, report = joblib.load(cache_path)
        write_json(report_dir / "report.json", report)
        return tables, report
    tables = {role: read_table(root / config["data"][role]["file"], config["data"][role], role == "labeled")
              for role in ["labeled", "unlabeled", "leaderboard"]}
    unique = pd.concat([frame.smiles for frame in tables.values()]).drop_duplicates().tolist()
    LOG.info("Canonicalizing %d distinct input SMILES", len(unique))
    # Repeated strings are parsed once; all source rows remain represented.
    structures = {smiles: canonicalize(smiles) for smiles in unique}
    report: dict[str, Any] = {"signature": signature, "manifest": manifest, "datasets": {}, "overlaps": {}}
    report_dir.mkdir(parents=True, exist_ok=True)
    for role, frame in tables.items():
        meta = pd.DataFrame([structures[s] for s in frame.smiles])
        frame = pd.concat([frame, meta], axis=1)
        tables[role] = frame
        invalid = frame.loc[frame.error.notna()]
        invalid.to_csv(report_dir / f"{role}_invalid.csv", index=False)
        duplicates = frame.loc[frame.structure_id.notna() & frame.structure_id.duplicated(keep=False)]
        duplicates.to_csv(report_dir / f"{role}_structure_duplicates.csv", index=False)
        frame.loc[frame.compound_id.duplicated(keep=False)].to_csv(report_dir / f"{role}_id_duplicates.csv", index=False)
        report["datasets"][role] = {
            "rows": len(frame), "invalid": len(invalid),
            "unique_structures": int(frame.structure_id.nunique()),
            "duplicate_structure_rows": len(duplicates),
            "duplicate_ids": int(frame.compound_id.duplicated().sum()),
        }
    labeled = tables["labeled"]
    conflicts = labeled.groupby("structure_id").label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    conflict_rows = labeled.loc[labeled.structure_id.isin(bad)]
    conflict_rows.to_csv(report_dir / "conflicting_labels.csv", index=False)
    report["conflicting_structures"] = len(bad)
    report["datasets"]["labeled"]["positives"] = int(labeled.label.sum())
    report["datasets"]["labeled"]["negatives"] = int((labeled.label == 0).sum())
    for left, right in itertools.combinations(tables, 2):
        details = {}
        for column in ["compound_id", "canonical_smiles"]:
            common = set(tables[left][column].dropna()) & set(tables[right][column].dropna())
            details[column] = len(common)
            pd.DataFrame({column: sorted(common)}).to_csv(report_dir / f"overlap_{left}_{right}_{column}.csv", index=False)
        report["overlaps"][f"{left}:{right}"] = details
    positive_scaffolds = labeled.loc[labeled.label == 1].groupby("scaffold").size().sort_values(ascending=False)
    report["positive_scaffold_counts"] = positive_scaffolds.to_dict()
    # IDs can be reused across sources; inconsistent mappings deserve an explicit diagnostic.
    all_rows = pd.concat([f.assign(source=role) for role, f in tables.items()], ignore_index=True)
    id_counts = all_rows.groupby("compound_id").structure_id.nunique()
    inconsistent = set(id_counts[id_counts > 1].index)
    all_rows.loc[all_rows.compound_id.isin(inconsistent)].to_csv(report_dir / "inconsistent_ids.csv", index=False)
    report["ids_mapping_to_multiple_structures"] = len(inconsistent)
    write_json(report_dir / "report.json", report)
    if any(report["datasets"][r]["invalid"] for r in ["labeled", "leaderboard"]):
        raise ValueError(f"Invalid labeled or leaderboard SMILES; see {report_dir}. No rows were dropped.")
    if bad and config["data"].get("fail_on_conflicts", True):
        raise ValueError(f"Conflicting activity labels on {len(bad)} structures; see {report_dir}")
    expected = config["data"].get("expected_leaderboard_rows")
    if expected is not None and len(tables["leaderboard"]) != expected:
        raise ValueError(f"Expected {expected} leaderboard rows, got {len(tables['leaderboard'])}")
    if config["submission"].get("unique_ids", True) and tables["leaderboard"].compound_id.duplicated().any():
        raise ValueError("Leaderboard compound IDs are not unique")
    joblib.dump((tables, report), cache_path, compress=3)
    LOG.info("Audit complete: %s", report["datasets"])
    return tables, report


def pretraining_entities(tables: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    """Return only chemical identity, never labels, to the self-supervised stage."""
    settings = config["pretraining"]
    mode = settings["mode"]
    if mode not in {"transductive", "inductive"}:
        raise ValueError("pretraining.mode must be transductive or inductive")
    frame = tables["unlabeled"].loc[:, ["structure_id", "canonical_smiles"]].dropna().drop_duplicates("structure_id")
    leaderboard = set(tables["leaderboard"].structure_id)
    excluded = set()
    if mode == "inductive":
        # Conservative sensitivity: no labeled or leaderboard structure is ever pretrained.
        excluded |= set(tables["labeled"].structure_id) | leaderboard
    elif not settings["include_leaderboard"]:
        excluded |= leaderboard
    frame = frame.loc[~frame.structure_id.isin(excluded)].sort_values("structure_id").reset_index(drop=True)
    maximum = settings.get("max_molecules")
    if maximum is not None and len(frame) > maximum:
        frame = frame.sample(n=int(maximum), random_state=config["seed"]).sort_values("structure_id").reset_index(drop=True)
    if len(frame) < 4:
        raise ValueError("Need at least four unique permitted structures for pretraining")
    return frame
