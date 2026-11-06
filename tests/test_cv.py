import numpy as np
import pandas as pd
import pytest

from dreamq.cross_validation import evaluate, fold_assignments, make_splits
from dreamq.metrics import cluster_bootstrap, evaluate_metrics, top_k_metrics
from dreamq.pretraining import pretrain


def test_group_splits_are_reproducible_and_disjoint():
    y = np.tile([0, 1], 20)
    groups = np.repeat(np.arange(20), 2)
    splits, _ = make_splits(y, groups, 5)
    repeated, _ = make_splits(y, groups, 5)
    for (train, valid), (again, _) in zip(splits, repeated):
        assert not set(groups[train]) & set(groups[valid])
        np.testing.assert_array_equal(train, again)
        assert set(y[valid]) == {0, 1}


def test_scaffold_folds_reduce_when_five_infeasible():
    y = np.array([1, 1, 0, 0, 0, 0])
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    splits, info = make_splits(y, groups, 5)
    assert len(splits) == 2 and info["limitation"]
    with pytest.raises(ValueError, match="No feasible"):
        make_splits(y, np.array(["positive", "positive", "a", "b", "c", "d"]), 5)


def test_fold_persistence_and_duplicate_grouping(config, molecules, tmp_path):
    duplicate = molecules.iloc[[0]].assign(compound_id="other_id")
    frame = pd.concat([molecules, duplicate], ignore_index=True)
    first, _, signature = fold_assignments(frame, "stratified", config, tmp_path)
    second, _, again = fold_assignments(frame, "stratified", config, tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert signature == again
    grouped = first.loc[first.structure_id == molecules.iloc[0].structure_id]
    assert grouped.fold.nunique() == 1


def test_ties_top_k_and_undefined_metrics():
    y = np.array([1, 0, 0, 0])
    ties = np.ones(4)
    assert top_k_metrics(y, ties, 1)["EF"] == 1
    assert top_k_metrics(y, ties, 1)["precision"] == .25
    assert evaluate_metrics(np.zeros(4), ties)["AP"] is None
    assert evaluate_metrics(np.ones(4), ties)["ROC_AUC"] is None
    assert evaluate_metrics(y, ties)["prevalence"] == .25
    assert cluster_bootstrap(y, ties, np.arange(4), 10)["valid_replicates"] <= 10


def test_nested_end_to_end_same_outer_splits(config, molecules, features, tmp_path):
    config["cv"]["strategies"] = ["stratified"]
    config["cv"]["repetitions"] = 2
    encoder = pretrain(features, config, tmp_path / "encoder")
    results = evaluate(molecules, features, encoder.transform(features), config, encoder.metadata)
    assert set(results.model) == {"M0", "M1", "M2", "M3", "similarity"}
    root = tmp_path / "artifacts" / "evaluation" / "stratified"
    oof = pd.read_csv(root / "oof_predictions.csv")
    pooled = pd.read_csv(root / "pooled_oof_predictions.csv")
    assert len(pooled) == 32 * 5
    assert oof.groupby(["model", "row_index"]).size().eq(2).all()
    assert oof.groupby(["row_index", "repetition"]).fold.nunique().eq(1).all()
