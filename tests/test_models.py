import numpy as np
import torch
import pytest

from dreamq.encoders import corrupt_bits
from dreamq.fusion import fuse_logits, select_alpha
from dreamq.models import LogitClassifier, SimilarityReference
from dreamq.pretraining import FrozenEncoder, pretrain


def test_reproducibility_serialization_and_raw_fingerprints(config, molecules, features, tmp_path):
    model = LogitClassifier().fit(features, molecules.label.to_numpy())
    matrix = model._matrix(features, None, fit=False)
    np.testing.assert_array_equal(matrix[:, :2048].toarray(), features.ecfp.toarray())
    model.save(tmp_path / "model.joblib")
    restored = LogitClassifier.load(tmp_path / "model.joblib")
    np.testing.assert_array_equal(model.predict_score(features), restored.predict_score(features))
    another = LogitClassifier().fit(features, molecules.label.to_numpy())
    np.testing.assert_allclose(model.predict_score(features), another.predict_score(features))


def test_classifier_preprocessing_has_no_validation_leakage(molecules, features):
    model = LogitClassifier().fit(features.take(np.arange(20)), molecules.label.to_numpy()[:20])
    mean = model.descriptor_transformer_.named_steps["scale"].mean_.copy()
    model.predict_score(features.take(np.arange(20, 32)))
    np.testing.assert_array_equal(mean, model.descriptor_transformer_.named_steps["scale"].mean_)
    np.testing.assert_allclose(mean, features.descriptors[:20].mean(axis=0), rtol=1e-6)


def test_similarity_uses_only_training_labels(molecules, features):
    baseline = SimilarityReference().fit(features.take(np.arange(20)), molecules.label.to_numpy()[:20])
    assert len(baseline.y_) == 20
    assert set(baseline.training_structure_ids_).isdisjoint(features.structure_ids[20:])
    assert np.isfinite(baseline.predict_score(features.take(np.arange(20, 32)))).all()


def test_encoder_training_checkpoint_and_freezing(config, features, tmp_path):
    encoder = pretrain(features, config, tmp_path / "encoder")
    assert encoder.transform(features).shape == (32, 128)
    assert all(not p.requires_grad for p in encoder.model.parameters())
    restored = FrozenEncoder.load(tmp_path / "encoder")
    np.testing.assert_allclose(encoder.transform(features), restored.transform(features), atol=1e-7)
    train_ids = set(encoder.metadata["training_structures"])
    valid_ids = set(encoder.metadata["validation_structures"])
    assert train_ids.isdisjoint(valid_ids)
    train_rows = [i for i, s in enumerate(features.structure_ids) if s in train_ids]
    np.testing.assert_allclose(encoder.normalization.named_steps["scale"].mean_, features.descriptors[train_rows].mean(axis=0), rtol=1e-6)


def test_fusion_and_corruption():
    a = np.array([-1, 1, -2, 2.])
    b = -a
    np.testing.assert_array_equal(fuse_logits(a, b, 0), a)
    alpha, history = select_alpha(np.array([0, 1, 0, 1]), a, b, [0, .5, 1])
    assert alpha == 0 and len(history) == 3
    np.testing.assert_array_equal(corrupt_bits(torch.ones((2, 10)), 1, 0).numpy(), 0)


def test_duplicate_pretraining_entities_rejected(config, features, tmp_path):
    with pytest.raises(ValueError, match="unique molecular structures"):
        pretrain(features.take([0, 0, 1, 2]), config, tmp_path / "encoder")
