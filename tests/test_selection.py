import numpy as np
import pytest
from scipy import sparse

from dreamq.representations import MolecularInputs
from dreamq.selection import select_diverse, selection_metrics


def fixture_inputs():
    return MolecularInputs(np.array(["b", "a", "c", "d"]), np.array(["C", "CC", "N", "O"]),
                           {"ecfp": sparse.csr_matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]], dtype=np.uint8)},
                           np.zeros((4, 20)), "test")


def test_top_k_and_novelty_tradeoff():
    inputs = fixture_inputs()
    scores = np.array([.9, .8, .7, .6])
    np.testing.assert_array_equal(select_diverse(inputs, scores, 2, 0), [0, 1])
    np.testing.assert_array_equal(select_diverse(inputs, scores, 2, 1), [0, 2])
    # Absolute score calibration does not alter percentile utility.
    np.testing.assert_array_equal(select_diverse(inputs, np.exp(scores), 2, 1), [0, 2])


def test_ties_are_deterministic_under_source_permutation():
    inputs = fixture_inputs()
    scores = np.ones(4)
    first = inputs.structure_ids[select_diverse(inputs, scores, 3)]
    permuted = inputs.take([2, 0, 3, 1])
    second = permuted.structure_ids[select_diverse(permuted, scores, 3)]
    np.testing.assert_array_equal(first, second)


def test_duplicate_structures_are_selected_only_once():
    inputs = fixture_inputs().take([0, 1, 2, 2])
    selected = select_diverse(inputs, np.array([.9, .8, .7, .6]), 3)
    assert len(set(inputs.structure_ids[selected])) == 3
    with pytest.raises(ValueError, match="unique molecular"):
        select_diverse(inputs, np.ones(4), 4)


def test_selection_validates_inputs_and_reports_hits():
    inputs = fixture_inputs()
    with pytest.raises(ValueError, match="finite"):
        select_diverse(inputs, np.array([.2, np.nan, .4, .5]), 2)
    with pytest.raises(ValueError, match="pool_size"):
        select_diverse(inputs, np.ones(4), 3, pool_size=2)
    with pytest.raises(ValueError, match="positive integer"):
        select_diverse(inputs, np.ones(4), 0)
    assert selection_metrics(np.array([1, 0, 1, 0]), np.array([0, 1])) == {
        "selected": 2, "hits": 1, "precision": .5, "recall": .5}
