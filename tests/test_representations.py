from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy import sparse

from dreamq.representations import featurize_inputs


def test_sparse_extended_features_and_true_counts(config, molecules, features, tmp_path):
    inputs = featurize_inputs(molecules, config["features"], config["cache_dir"], features)
    assert inputs.matrices["counts"].dtype == np.uint32
    assert inputs.matrices["counts"].max() > 1
    assert inputs.matrices["maccs"].shape == (32, 167)
    assert inputs.matrices["atompair"].shape == (32, 2048)
    assert inputs.matrices["rdkit"].shape == (32, 2048)
    assert inputs.matrices["avalon"].shape == (32, 512)
    for name, matrix in inputs.matrices.items():
        assert sparse.isspmatrix_csr(matrix)
        if name != "counts":
            assert np.isin(matrix.data, [1]).all()
    np.testing.assert_array_equal(inputs.structure_ids, molecules.structure_id)
    np.testing.assert_array_equal(inputs.matrices["ecfp"].toarray(), features.ecfp.toarray())
    assert not hasattr(inputs, "labels")
    reordered = molecules.iloc[[5, 2, 5]]
    selected = inputs.for_frame(reordered)
    np.testing.assert_array_equal(selected.structure_ids, reordered.structure_id)
    joblib.dump(inputs, tmp_path / "inputs.joblib")
    restored = joblib.load(tmp_path / "inputs.joblib")
    np.testing.assert_array_equal(restored.take([2]).matrices["counts"].toarray(), inputs.take([2]).matrices["counts"].toarray())


def test_supplementary_cache_is_order_independent(config, molecules, features):
    original = featurize_inputs(molecules, config["features"], config["cache_dir"], features)
    shuffled = molecules.iloc[::-1]
    again = featurize_inputs(shuffled, config["features"], config["cache_dir"], features)
    assert original.signature == again.signature
    for name in original.matrices:
        np.testing.assert_array_equal(original.matrices[name].toarray(), again.matrices[name][::-1].toarray())
    assert len(list((Path(config["cache_dir"]) / "supplementary_features").glob("*.joblib"))) == 1


def test_inconsistent_identity_and_missing_base_are_rejected(config, molecules, features):
    invalid = molecules.copy()
    invalid.loc[0, "canonical_smiles"] = "N"
    with pytest.raises(ValueError, match="hashes"):
        featurize_inputs(invalid, config["features"], config["cache_dir"], features)
    with pytest.raises(ValueError, match="lack"):
        featurize_inputs(molecules, config["features"], config["cache_dir"], features.take([0, 1]))
