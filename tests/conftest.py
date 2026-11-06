from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dreamq.data import canonicalize
from dreamq.features import featurize
from dreamq.utils import read_config


@pytest.fixture
def config(tmp_path):
    config = read_config(Path(__file__).parents[1] / "configs/default.yaml")
    for key, name in [("data_dir", "data"), ("cache_dir", "cache"), ("artifact_dir", "artifacts"), ("prediction_dir", "predictions")]:
        config[key] = str(tmp_path / name)
        Path(config[key]).mkdir()
    config["data"]["expected_leaderboard_rows"] = 2
    config["features"]["n_jobs"] = 1
    config["models"].update(C=[0.1], positive_weight=[1], max_iter=1000)
    config["pretraining"].update(max_epochs=1, batch_size=8, max_molecules=None, device="cpu")
    config["cv"].update(folds=2, inner_folds=2, repetitions=1, bootstrap_samples=5)
    config["threads"] = 1
    return config


@pytest.fixture
def molecules():
    # Chemically distinct small fixtures, not a scientific activity dataset.
    smiles = ["C" * n + suffix for suffix in ["O", "N", "C(=O)O", "Cl"] for n in range(1, 9)]
    frame = pd.DataFrame({"compound_id": [f"m{i}" for i in range(len(smiles))], "smiles": smiles,
                          "label": np.tile([0, 0, 0, 1], len(smiles) // 4), "row_index": np.arange(len(smiles))})
    return pd.concat([frame, pd.DataFrame([canonicalize(s) for s in smiles])], axis=1)


@pytest.fixture
def features(config, molecules):
    return featurize(molecules, config["features"], config["cache_dir"]).for_frame(molecules)


def write_dataset(config, molecules, conflicting=False, invalid_leaderboard=False):
    root = Path(config["data_dir"])
    selected = molecules.iloc[:16].copy()
    duplicate = selected.iloc[[0]].copy()
    duplicate["compound_id"] = "duplicate"
    if conflicting:
        duplicate["label"] = 1 - duplicate.label
    selected = pd.concat([selected, duplicate], ignore_index=True)
    pd.DataFrame({"CatalogID": selected.compound_id, "RDKit_SMILES": selected.smiles, "Label": selected.label}).to_csv(root / config["data"]["labeled"]["file"], index=False)
    pd.DataFrame({"CatalogID": molecules.compound_id, "SMILES": molecules.smiles, "Label": ["MUST_NOT_READ"] * len(molecules)}).to_csv(root / config["data"]["unlabeled"]["file"], index=False)
    pd.DataFrame({"CatalogID": ["leader1", "leader2"], "RDKit_SMILES": ["invalid" if invalid_leaderboard else "c1ccccc1", "c1ccncc1"],
                  "Label": ["HIDDEN", "HIDDEN"]}).to_csv(root / config["data"]["leaderboard"]["file"], index=False)
    entries = []
    for role in ["labeled", "unlabeled", "leaderboard"]:
        filename = config["data"][role]["file"]
        entries.append({"name": filename, "dataFileMD5Hex": hashlib.md5((root / filename).read_bytes()).hexdigest()})
    pd.DataFrame(entries).to_csv(root / "manifest.csv", index=False)
