"""Final refitting, submission validation, and future residual-learner interfaces."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from .cross_validation import fit_branches, make_splits
from .data import canonicalize
from .features import FeatureBundle, feature_signature, molecule_features
from .fusion import fuse_logits
from .models import LogitClassifier
from .pretraining import FrozenEncoder
from .utils import digest, versions, write_json


class ClassicalInterface(Protocol):
    """Contract for a future residual learner; no residual model is implemented."""
    def classical_logit(self, molecule: str) -> float: ...
    def molecular_embedding(self, molecule: str) -> np.ndarray: ...
    def pharmacophore_features(self, molecule: str) -> sparse.csr_matrix: ...
    def predict_score(self, molecule: str) -> float: ...


def training_signature(frame: pd.DataFrame) -> str:
    return digest(frame[["compound_id", "structure_id", "label"]].values.tolist())


class MolecularPredictor:
    def __init__(self, branch_a: LogitClassifier, branch_b: LogitClassifier, encoder: FrozenEncoder,
                 alpha: float, feature_config: dict, mode: str = "M3", metadata: dict | None = None):
        if mode not in {"M1", "M2", "M3"}:
            raise ValueError("Prediction mode must be M1, M2, or M3")
        self.branch_a, self.branch_b, self.encoder = branch_a, branch_b, encoder
        self.alpha, self.feature_config, self.mode = alpha, feature_config, mode
        self.metadata = metadata or {}

    def _features(self, molecule: str) -> FeatureBundle:
        identity = canonicalize(molecule)
        if identity["error"]:
            raise ValueError(f"Invalid SMILES: {molecule!r}")
        ecfp, pharm, descriptors = molecule_features(identity["canonical_smiles"], self.feature_config)
        def row(indices, size):
            return sparse.csr_matrix((np.ones(len(indices), dtype=np.uint8), indices, [0, len(indices)]), shape=(1, size))
        return FeatureBundle(np.array([identity["structure_id"]]), row(ecfp, self.feature_config["morgan_bits"]),
                             row(pharm, self.feature_config["pharmacophore_bits"]), descriptors[None, :], feature_signature(self.feature_config))

    def predict_features(self, features: FeatureBundle, mode: str | None = None) -> dict[str, np.ndarray]:
        embeddings = self.encoder.transform(features)
        z_a = self.branch_a.decision_function(features)
        z_b = self.branch_b.decision_function(features, embeddings)
        z_fused = fuse_logits(z_a, z_b, self.alpha)
        selected_mode = mode or self.mode
        if selected_mode not in {"M1", "M2", "M3"}:
            raise ValueError("Prediction mode must be M1, M2, or M3")
        classical = {"M1": z_a, "M2": z_b, "M3": z_fused}[selected_mode]
        return {"branch_a_logit": z_a, "branch_b_logit": z_b, "fused_logit": z_fused,
                "classical_logit": classical, "activity_score": expit(classical), "embedding": embeddings}

    def classical_logit(self, molecule: str) -> float:
        return float(self.predict_features(self._features(molecule))["classical_logit"][0])

    def molecular_embedding(self, molecule: str) -> np.ndarray:
        return self.encoder.transform(self._features(molecule))[0]

    def pharmacophore_features(self, molecule: str) -> sparse.csr_matrix:
        return self._features(molecule).pharmacophore

    def predict_score(self, molecule: str) -> float:
        return float(expit(self.classical_logit(molecule)))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Self-contained classical artifact including the exact frozen encoder.
        encoder = FrozenEncoder(deepcopy(self.encoder.model).cpu(), self.encoder.normalization, self.encoder.metadata, "cpu")
        artifact = MolecularPredictor(self.branch_a, self.branch_b, encoder, self.alpha, self.feature_config, self.mode, self.metadata)
        joblib.dump(artifact, directory / "predictor.joblib", compress=3)
        write_json(directory / "metadata.json", self.metadata)

    @staticmethod
    def load(directory: str | Path) -> MolecularPredictor:
        return joblib.load(Path(directory) / "predictor.joblib")


def fit_final(frame: pd.DataFrame, features: FeatureBundle, encoder: FrozenEncoder, config: dict) -> MolecularPredictor:
    embeddings = encoder.transform(features)
    y = frame.label.to_numpy()
    strategy = config["cv"].get("final_strategy", "stratified")
    if strategy not in {"stratified", "scaffold"}:
        raise ValueError("final_strategy must be stratified or scaffold")
    groups = frame["structure_id" if strategy == "stratified" else "scaffold"].to_numpy()
    splits, info = make_splits(y, groups, config["cv"]["inner_folds"], config["seed"], config["cv"]["split_attempts"])
    models, alpha, selection, inner_logits = fit_branches(features, y, embeddings, splits, config, config["seed"], include_m0=False)
    metadata = {"selection": selection, "splits": info, "final_strategy": strategy, "config": config,
                "versions": versions(), "encoder_identity": encoder.metadata["identity"],
                "training_data_signature": training_signature(frame), "feature_signature": features.signature,
                "training_structure_ids": features.structure_ids.tolist(),
                "development": config.get("development", False), "calibrated": False}
    predictor = MolecularPredictor(models["M1"], models["M2"], encoder, alpha, config["features"], metadata=metadata)
    directory = Path(config["artifact_dir"]) / "final"
    predictor.save(directory)
    pd.DataFrame({"row_index": np.arange(len(frame)), **{f"logit_{k}": v for k, v in inner_logits.items()}}).to_csv(directory / "inner_oof.csv", index=False)
    pd.DataFrame([{"fold": j, "row_index": int(i)} for j, (_, val) in enumerate(splits) for i in val]).to_csv(directory / "inner_folds.csv", index=False)
    return predictor


def validate_submission(submission: pd.DataFrame, expected_ids: list[str], settings: dict, expected_count: int | None = None) -> None:
    id_column, score_column = settings["id_column"], settings["score_column"]
    if id_column == score_column:
        raise ValueError("Submission ID and score column names must differ")
    if list(submission.columns) != [id_column, score_column]:
        raise ValueError("Submission column names/order do not match configuration")
    if submission[id_column].tolist() != expected_ids:
        raise ValueError("Submission compound IDs or row ordering differ from source")
    if expected_count is not None and len(submission) != expected_count:
        raise ValueError(f"Expected {expected_count} submission rows, got {len(submission)}")
    if settings.get("unique_ids", True) and submission[id_column].duplicated().any():
        raise ValueError("Duplicate submission compound IDs")
    scores = submission[score_column].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Submission scores must be finite and in [0, 1]")


def predict_table(frame: pd.DataFrame, features: FeatureBundle, predictor: MolecularPredictor, config: dict,
                  mode: str | None = None, official: bool = True) -> pd.DataFrame:
    if frame.canonical_smiles.isna().any():
        raise ValueError("Invalid molecules in prediction set; no partial submission will be written")
    output = predictor.predict_features(features, mode)
    settings = config["submission"]
    submission = pd.DataFrame({settings["id_column"]: frame.compound_id.to_numpy(), settings["score_column"]: output["activity_score"]})
    validate_submission(submission, frame.compound_id.tolist(), settings,
                        config["data"].get("expected_leaderboard_rows") if official else len(frame))
    directory = Path(config["prediction_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    submission.to_csv(directory / (settings["file"] if official else "arbitrary_scores.csv"), index=False)
    diagnostics = pd.DataFrame({"compound_id": frame.compound_id.to_numpy(), "structure_id": frame.structure_id.to_numpy(),
                               **{key: values for key, values in output.items() if key != "embedding"}})
    diagnostics = pd.concat([diagnostics, pd.DataFrame(output["embedding"], columns=[f"h_{i:03d}" for i in range(output["embedding"].shape[1])])], axis=1)
    diagnostics.to_csv(directory / (settings["diagnostic_file"] if official else "arbitrary_diagnostics.csv"), index=False)
    write_json(directory / "prediction_metadata.json", {"rows": len(frame), "mode": mode or predictor.mode,
               "alpha": predictor.alpha, "development": config.get("development", False),
               "encoder_identity": predictor.encoder.metadata["identity"], "score_calibrated": False,
               "submission_format_verified_against_challenge_instructions": False,
               "format_note": "Local manifest specifies files, not the official submission schema. Column names are user-configurable."})
    return submission
