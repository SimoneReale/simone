"""Command-line orchestration; data, feature cache, models, and scores stay separate."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .cross_validation import evaluate
from .data import audit, canonicalize, pretraining_entities, read_table
from .features import feature_signature, featurize
from .inference import MolecularPredictor, fit_final, predict_table, training_signature
from .pretraining import FrozenEncoder, pretrain
from .utils import configure_logging, digest, read_config, seed_all, write_json

LOG = logging.getLogger(__name__)


def prepare(config: dict):
    tables, report = audit(config)
    permitted = pretraining_entities(tables, config)
    entities = pd.concat([permitted, tables["labeled"][["structure_id", "canonical_smiles"]],
                          tables["leaderboard"][["structure_id", "canonical_smiles"]]], ignore_index=True)
    features = featurize(entities, config["features"], config["cache_dir"])
    return tables, report, permitted, features


def load_encoder(config: dict, permitted: pd.DataFrame, feature_signature: str) -> FrozenEncoder:
    directory = Path(config["artifact_dir"]) / "encoder"
    if not (directory / "encoder.pt").exists():
        raise FileNotFoundError("No pretrained encoder. Run the pretrain command first.")
    encoder = FrozenEncoder.load(directory)
    expected = digest({"features": sorted(permitted.structure_id.tolist()), "config": config["pretraining"],
                       "feature_signature": feature_signature, "seed": config["seed"]})
    if encoder.metadata["identity"] != expected:
        raise ValueError("Encoder provenance differs from current data/configuration; rerun pretrain")
    return encoder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "featurize", "pretrain", "evaluate", "fit-final", "predict", "run", "list-models", "compare", "fit-models", "predict-models", "run-models"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dev", action="store_true", help="Reduced pretraining/grid/folds; outputs are explicitly marked development")
    parser.add_argument("--run-name", help="Separate output name for comparison experiments")
    parser.add_argument("--models", help="Comma-separated independent model IDs for comparison commands, or all")
    parser.add_argument("--mode", choices=["M1", "M2", "M3"], default=None)
    parser.add_argument("--input", help="Arbitrary compound CSV for prediction/export")
    parser.add_argument("--id-column", default="CatalogID")
    parser.add_argument("--smiles-column", default="SMILES")
    args = parser.parse_args(argv)
    config = read_config(args.config, args.dev)
    comparison_command = args.command in {"list-models", "compare", "fit-models", "predict-models", "run-models"}
    if args.models and not comparison_command:
        parser.error("--models applies to comparison commands")
    if args.run_name:
        if not comparison_command:
            parser.error("--run-name applies to comparison commands")
        config.setdefault("zoo", {})["output_name"] = args.run_name
    configure_logging(Path(config["artifact_dir"]) / "logs" / f"{args.command}.jsonl")
    seed_all(config["seed"], config.get("threads", 2))
    Path(config["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    if not comparison_command:
        write_json(Path(config["artifact_dir"]) / "resolved_config.json", config)
    if args.command in {"list-models", "compare", "fit-models", "predict-models", "run-models"}:
        from .experiments import (catalog, requested_models, required_models, prepare_inputs, compare_models,
                                  fit_final_models, ComparisonPredictor, predict_models, comparison_root)
        if args.command == "list-models":
            for name, spec in catalog().items():
                suffix = " (optional dependency missing)" if spec.get("installed") is False else ""
                print(f"{name:12} {spec['label']}{suffix}")
            return
        names = requested_models(config, args.models)
        arbitrary = None
        if args.input:
            if args.command != "predict-models":
                parser.error("--input is supported with predict-models or predict")
            arbitrary = read_table(Path(args.input), {"id": args.id_column, "smiles": args.smiles_column}, labeled=False)
            arbitrary = pd.concat([arbitrary, pd.DataFrame([canonicalize(s) for s in arbitrary.smiles])], axis=1)
        predictor = None
        if args.command == "predict-models":
            predictor = ComparisonPredictor.load(config)
            names = requested_models(config, args.models or predictor.requested)
            if not set(names) <= set(predictor.models):
                raise ValueError("Requested models are absent from final checkpoint")
        required_models(names, config)  # Fail fast before expensive data extraction.
        tables, inputs, report, encoder = prepare_inputs(config, names, arbitrary, predictor.encoder if predictor else None)
        labeled = inputs.for_frame(tables["labeled"])
        if predictor is not None:
            if predictor.metadata["training_data_signature"] != training_signature(tables["labeled"]):
                raise ValueError("Labeled data differs from fitted comparison models")
            if encoder is not None and encoder.metadata["settings"] != config["pretraining"]:
                raise ValueError("Requested pretraining protocol differs from fitted comparison models")
        write_json(comparison_root(config) / "resolved_config.json", config)
        if args.command in {"compare", "run-models"}:
            results = compare_models(tables["labeled"], labeled, config, names, encoder.metadata if encoder else None)
            LOG.info("Independent model comparison:\n%s", results.to_string(index=False))
        if args.command in {"fit-models", "run-models"}:
            predictor = fit_final_models(tables["labeled"], labeled, config, names, encoder)
        if args.command in {"predict-models", "run-models"}:
            frame = tables["leaderboard"] if arbitrary is None else arbitrary
            predict_models(frame, inputs.for_frame(frame), predictor, config, names, official=arbitrary is None)
            LOG.info("Wrote %d rows for %d models under %s/comparison", len(frame), len(names), config["prediction_dir"])
        return
    if args.command == "audit":
        audit(config)
        return
    if args.command == "predict":
        tables, _ = audit(config)
        predictor = MolecularPredictor.load(Path(config["artifact_dir"]) / "final")
        if feature_signature(config["features"]) != feature_signature(predictor.feature_config):
            raise ValueError("Prediction feature configuration differs from fitted model")
        if predictor.metadata["training_data_signature"] != training_signature(tables["labeled"]):
            raise ValueError("Labeled dataset differs from the fitted predictor; refit before prediction")
        if predictor.encoder.metadata["settings"] != config["pretraining"]:
            raise ValueError("Requested pretraining protocol differs from fitted predictor")
        frame = tables["leaderboard"]
        if args.input:
            frame = read_table(Path(args.input), {"id": args.id_column, "smiles": args.smiles_column}, labeled=False)
            frame = pd.concat([frame, pd.DataFrame([canonicalize(s) for s in frame.smiles])], axis=1)
        features = featurize(frame, config["features"], config["cache_dir"]).for_frame(frame)
        submission = predict_table(frame, features, predictor, config, args.mode, official=not bool(args.input))
        LOG.info("Wrote %d validated predictions to %s", len(submission), config["prediction_dir"])
        return
    tables, report, permitted, features = prepare(config)
    if args.command == "featurize":
        return
    if args.command in {"pretrain", "run"}:
        permitted_features = features.for_frame(permitted)
        # Keep pretrained entity ordering deterministic and part of checkpoint identity.
        encoder = pretrain(permitted_features, config, Path(config["artifact_dir"]) / "encoder",
                           {"audit_signature": report["signature"], "mode": config["pretraining"]["mode"],
                            "leaderboard_structures_in_pretraining": int(permitted.structure_id.isin(tables["leaderboard"].structure_id).sum()),
                            "labeled_structures_in_pretraining": int(permitted.structure_id.isin(tables["labeled"].structure_id).sum())})
        if args.command == "pretrain":
            return
    else:
        encoder = load_encoder(config, permitted, features.signature)
    labeled_features = features.for_frame(tables["labeled"])
    if args.command in {"evaluate", "run"}:
        results = evaluate(tables["labeled"], labeled_features, encoder.transform(labeled_features), config, encoder.metadata)
        LOG.info("Evaluation results:\n%s", results.to_string(index=False))
        if args.command == "evaluate":
            return
    if args.command in {"fit-final", "run"}:
        predictor = fit_final(tables["labeled"], labeled_features, encoder, config)
        LOG.info("Final refit complete (development=%s)", config.get("development", False))
    if args.command == "run":
        predict_table(tables["leaderboard"], features.for_frame(tables["leaderboard"]), predictor, config, args.mode)


if __name__ == "__main__":
    main()
