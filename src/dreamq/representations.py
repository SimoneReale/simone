"""Sparse, label-free inputs for the extended molecular model comparison.

The original feature representation and cache remain unchanged. Supplementary
Morgan counts, MACCS, Atom Pair, RDKit-path and Avalon fingerprints have their
own versioned cache. MolecularInputs contains no activity labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from scipy import sparse

from .features import FeatureBundle, feature_signature, featurize
from .utils import digest

LOG = logging.getLogger(__name__)
REPRESENTATION_VERSION = 1
EXTRA_WIDTHS = {"maccs": 167, "atompair": 2048, "rdkit": 2048, "avalon": 512}


@dataclass
class MolecularInputs:
    structure_ids: np.ndarray
    canonical_smiles: np.ndarray
    matrices: dict[str, sparse.csr_matrix]
    descriptors: np.ndarray
    signature: str
    embeddings: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.structure_ids)

    def take(self, indices: np.ndarray | list[int]) -> MolecularInputs:
        idx = np.asarray(indices, dtype=int)
        return MolecularInputs(self.structure_ids[idx], self.canonical_smiles[idx],
                               {name: matrix[idx] for name, matrix in self.matrices.items()},
                               self.descriptors[idx], self.signature,
                               None if self.embeddings is None else self.embeddings[idx])

    def for_frame(self, frame: pd.DataFrame) -> MolecularInputs:
        lookup = {identifier: i for i, identifier in enumerate(self.structure_ids)}
        missing = set(frame.structure_id) - set(lookup)
        if missing:
            raise ValueError(f"Feature inputs lack {len(missing)} requested molecular structures")
        result = self.take([lookup[identifier] for identifier in frame.structure_id])
        if "canonical_smiles" in frame and not np.array_equal(result.canonical_smiles, frame.canonical_smiles.to_numpy()):
            raise ValueError("Structure identifiers map to inconsistent canonical SMILES")
        return result


def representation_signature(feature_config: dict) -> str:
    return digest({"version": REPRESENTATION_VERSION, "rdkit": rdBase.rdkitVersion,
                   "legacy": feature_signature(feature_config), "widths": EXTRA_WIDTHS,
                   "counts": "Morgan GetCountFingerprint; uint32, no count simulation",
                   "atompair": "GetAtomPairGenerator default binary count simulation",
                   "maccs": "RDKit 167 positions, unused bit zero retained"})


def _extra_features(smiles: str, radius: int, bits: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        from rdkit.Avalon import pyAvalonTools
    except ImportError as error:
        raise ImportError("Extended fingerprints require an RDKit build with Avalon support") from error
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid molecular structure in supplementary features: {smiles!r}")
    counts = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits).GetCountFingerprint(mol).GetNonzeroElements()
    positions = np.array(sorted(counts), dtype=np.int32)
    values = np.array([counts[int(i)] for i in positions], dtype=np.uint32)
    result = {"counts": (positions, values)}
    fingerprints = {
        "maccs": MACCSkeys.GenMACCSKeys(mol),
        "atompair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprint(mol),
        "rdkit": Chem.RDKFingerprint(mol, fpSize=2048),
        "avalon": pyAvalonTools.GetAvalonFP(mol, nBits=512),
    }
    for name, fingerprint in fingerprints.items():
        on_bits = np.asarray(fingerprint.GetOnBits(), dtype=np.int32)
        result[name] = (on_bits, np.ones(len(on_bits), dtype=np.uint8))
    return result


def _sparse_rows(rows: list[tuple[np.ndarray, np.ndarray]], width: int, dtype) -> sparse.csr_matrix:
    pointers = np.cumsum([0] + [len(indices) for indices, _ in rows], dtype=np.int64)
    positions = np.concatenate([indices for indices, _ in rows])
    values = np.concatenate([values for _, values in rows]).astype(dtype, copy=False)
    return sparse.csr_matrix((values, positions, pointers), shape=(len(rows), width), dtype=dtype)


def featurize_inputs(frame: pd.DataFrame, feature_config: dict, cache_dir: str | Path,
                     base: FeatureBundle | None = None) -> MolecularInputs:
    """Extract unique structures once and return inputs in the requested row order.

    Labels and other source columns are not persisted. True Morgan counts use
    uint32 rather than uint8; binary fingerprint bits are never standardized.
    """
    if not {"structure_id", "canonical_smiles"} <= set(frame.columns):
        raise ValueError("Extended features require audited structure_id and canonical_smiles columns")
    if len(frame) == 0 or frame[["structure_id", "canonical_smiles"]].isna().any().any():
        raise ValueError("Extended features require nonempty, valid audited molecules")
    entities = frame.loc[:, ["structure_id", "canonical_smiles"]].drop_duplicates().sort_values("structure_id")
    if entities.structure_id.duplicated().any():
        raise ValueError("Structure identifiers map to inconsistent canonical SMILES")
    if any(hashlib.sha256(str(s).encode()).hexdigest() != identifier
           for identifier, s in entities.itertuples(index=False, name=None)):
        raise ValueError("Structure identifiers do not match their canonical SMILES hashes")
    if base is None:
        base = featurize(entities, feature_config, cache_dir)
    if base.signature != feature_signature(feature_config):
        raise ValueError("Base feature configuration differs from requested supplementary features")
    try:
        aligned_base = base.for_frame(entities)
    except KeyError as error:
        raise ValueError("Base features lack requested molecular structures") from error
    signature = representation_signature(feature_config)
    key = digest({"signature": signature, "structures": entities.values.tolist()})
    cache = Path(cache_dir) / "supplementary_features"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{key}.joblib"
    if target.exists():
        LOG.info("Loading supplementary fingerprints from %s", target)
        extra = joblib.load(target)
    else:
        collected: dict[str, list] = {name: [] for name in ["counts", *EXTRA_WIDTHS]}
        smiles = entities.canonical_smiles.tolist()
        with joblib.Parallel(n_jobs=feature_config.get("n_jobs", 1)) as parallel:
            for start in range(0, len(smiles), 1024):
                batch = parallel(joblib.delayed(_extra_features)(s, feature_config["morgan_radius"], feature_config["morgan_bits"])
                                 for s in smiles[start:start + 1024])
                for result in batch:
                    for name, value in result.items():
                        collected[name].append(value)
                LOG.info("Supplementary fingerprints: %d / %d", min(start + 1024, len(smiles)), len(smiles))
        extra = {name: _sparse_rows(rows, feature_config["morgan_bits"] if name == "counts" else EXTRA_WIDTHS[name],
                                   np.uint32 if name == "counts" else np.uint8) for name, rows in collected.items()}
        temporary = target.with_suffix(f".{uuid4().hex}.tmp")
        joblib.dump(extra, temporary, compress=3)
        temporary.replace(target)
    inputs = MolecularInputs(entities.structure_id.to_numpy(), entities.canonical_smiles.to_numpy(),
                             {"ecfp": aligned_base.ecfp, "pharmacophore": aligned_base.pharmacophore, **extra},
                             aligned_base.descriptors, signature)
    return inputs.for_frame(frame)
