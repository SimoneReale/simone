from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dreamq.cli import main
from dreamq.inference import MolecularPredictor, fit_final, predict_table, validate_submission
from dreamq.pretraining import pretrain
from conftest import write_dataset


def test_final_refit_prediction_and_interface(config, molecules, features, tmp_path):
    encoder = pretrain(features, config, tmp_path / "encoder")
    predictor = fit_final(molecules, features, encoder, config)
    frame = molecules.iloc[:2].copy()
    submission = predict_table(frame, features.take([0, 1]), predictor, config)
    assert submission.CatalogID.tolist() == frame.compound_id.tolist()
    assert list(submission.columns) == ["CatalogID", "ActivityScore"]
    assert len(pd.read_csv(Path(config["prediction_dir"], "leaderboard_diagnostics.csv")).filter(regex="^h_" ).columns) == 128
    restored = MolecularPredictor.load(Path(config["artifact_dir"]) / "final")
    np.testing.assert_allclose(predictor.predict_features(features)["activity_score"], restored.predict_features(features)["activity_score"])
    assert predictor.molecular_embedding("CCO").shape == (128,)
    assert predictor.pharmacophore_features("CCO").shape == (1, 1024)
    assert 0 <= predictor.predict_score("CCO") <= 1
    assert np.isfinite(predictor.classical_logit("CCO"))
    for mode in ["M1", "M2", "M3"]:
        assert np.isfinite(predictor.predict_features(features, mode)["activity_score"]).all()


def test_bad_submission_rejected(config):
    settings = config["submission"]
    with pytest.raises(ValueError, match="finite"):
        validate_submission(pd.DataFrame({"CatalogID": ["x"], "ActivityScore": [np.nan]}), ["x"], settings)
    with pytest.raises(ValueError, match="ordering"):
        validate_submission(pd.DataFrame({"CatalogID": ["y"], "ActivityScore": [.5]}), ["x"], settings)
    with pytest.raises(ValueError, match="Expected"):
        validate_submission(pd.DataFrame({"CatalogID": ["x"], "ActivityScore": [.5]}), ["x"], settings, 300)


def test_cli_all_stages_synthetic_csv(config, molecules, tmp_path):
    write_dataset(config, molecules)
    config["cv"]["strategies"] = ["stratified"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    main(["run", "--config", str(path)])
    submission = pd.read_csv(Path(config["prediction_dir"], "submission.csv"))
    assert submission.CatalogID.tolist() == ["leader1", "leader2"]
    # Exercises reloading with the activity-free checkpoint identity check.
    main(["fit-final", "--config", str(path)])
    main(["predict", "--config", str(path), "--mode", "M1"])
    config["pretraining"]["mode"] = "inductive"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="protocol differs"):
        main(["predict", "--config", str(path)])
