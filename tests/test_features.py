import numpy as np
from scipy import sparse
import pytest

from dreamq.features import CompactProjector, DESCRIPTORS, descriptor_transformer, feature_signature, molecule_features


def test_dimensions_sparse_and_determinism(features, config):
    assert features.ecfp.shape == (32, 2048)
    assert features.pharmacophore.shape == (32, 1024)
    assert features.descriptors.shape == (32, 20)
    assert sparse.isspmatrix_csr(features.ecfp)
    assert features.ecfp.dtype == np.uint8
    first = molecule_features("CCO", config["features"])
    second = molecule_features("OCC", config["features"])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    # A molecule with several pharmacophore feature families must have active bits.
    _, pharm, _ = molecule_features("CC(=O)Nc1ccc(C(=O)O)cc1", config["features"])
    assert len(pharm) > 0
    assert pharm.min() >= 0 and pharm.max() < 1024


def test_cache_configuration_invalidation(config):
    signature = feature_signature(config["features"])
    config["features"]["morgan_radius"] = 3
    assert feature_signature(config["features"]) != signature


def test_imputation_scaling_fit_only_training():
    train = np.array([[1., np.nan], [3., np.nan]])
    transform = descriptor_transformer().fit(train)
    mean = transform.named_steps["scale"].mean_.copy()
    result = transform.transform([[99999., 4.]])
    np.testing.assert_array_equal(mean, [2., 0.])
    np.testing.assert_array_equal(transform.named_steps["scale"].mean_, mean)
    assert np.isfinite(result).all() and result.shape == (1, 2)


@pytest.mark.parametrize("dimensions", [4, 6, 8])
def test_compact_projection_training_only(features, dimensions):
    projector = CompactProjector(dimensions).fit(features.pharmacophore[:20])
    result = projector.transform(features.pharmacophore[20:])
    assert result.shape == (12, dimensions)
    assert np.all(result >= 0) and np.all(result <= np.pi + 1e-7)


def test_invalid_smiles_features_raise(config):
    with pytest.raises(ValueError, match="invalid SMILES"):
        molecule_features("invalid", config["features"])
