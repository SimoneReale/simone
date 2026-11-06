from pathlib import Path

import pandas as pd
import pytest

from dreamq.data import audit, canonicalize, pretraining_entities
from dreamq.data import read_table
from conftest import write_dataset


def test_canonicalization_and_invalid():
    assert canonicalize("OCC")["structure_id"] == canonicalize("CCO")["structure_id"]
    assert canonicalize("not-a-molecule")["error"]
    assert canonicalize("")["error"]
    assert canonicalize("[Na+].[Cl-]")["canonical_smiles"]
    assert canonicalize("C[C@H](O)N")["structure_id"] != canonicalize("C[C@@H](O)N")["structure_id"]


def test_audit_duplicates_overlaps_and_hidden_labels(config, molecules):
    write_dataset(config, molecules)
    tables, report = audit(config)
    assert len(tables["labeled"]) == 17  # duplicate rows explicitly preserved
    assert report["datasets"]["labeled"]["duplicate_structure_rows"] == 2
    assert report["overlaps"]["labeled:unlabeled"]["canonical_smiles"] == 16
    assert "label" not in tables["unlabeled"]
    assert "Label" not in tables["leaderboard"]
    assert "label" not in tables["leaderboard"]
    assert Path(config["artifact_dir"], "audit", "report.json").exists()
    _, cached = audit(config)
    assert cached == report


def test_conflicting_labels_fail_and_persist_report(config, molecules):
    write_dataset(config, molecules, conflicting=True)
    with pytest.raises(ValueError, match="Conflicting"):
        audit(config)
    assert Path(config["artifact_dir"], "audit", "conflicting_labels.csv").exists()


def test_invalid_leaderboard_never_silently_dropped(config, molecules):
    write_dataset(config, molecules, invalid_leaderboard=True)
    with pytest.raises(ValueError, match="Invalid labeled or leaderboard"):
        audit(config)
    assert len(pd.read_csv(Path(config["artifact_dir"], "audit", "leaderboard_invalid.csv"))) == 1


def test_manifest_tampering_fails(config, molecules):
    write_dataset(config, molecules)
    with Path(config["data_dir"], config["data"]["labeled"]["file"]).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        audit(config)


def test_inductive_pretraining_excludes_all_supervised_and_leaderboard(config, molecules):
    write_dataset(config, molecules)
    tables, _ = audit(config)
    config["pretraining"].update(mode="inductive", include_leaderboard=False)
    permitted = pretraining_entities(tables, config)
    assert not set(permitted.structure_id) & set(tables["labeled"].structure_id)
    assert not set(permitted.structure_id) & set(tables["leaderboard"].structure_id)
    assert set(permitted.columns) == {"structure_id", "canonical_smiles"}


def test_unlabeled_rows_are_not_assigned_negative_activity(config, molecules):
    write_dataset(config, molecules)
    tables, _ = audit(config)
    assert len(pretraining_entities(tables, config)) == len(molecules)
    assert set(tables["labeled"].label) == {0, 1}


def test_missing_smiles_produces_clear_invalid_diagnostic(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"id": ["a", "b"], "smi": [None, "CCO"]}).to_csv(path, index=False)
    frame = read_table(path, {"id": "id", "smiles": "smi"}, labeled=False)
    assert canonicalize(frame.iloc[0].smiles)["error"] == "empty SMILES"
