import numpy as np

from rfbdd.discretize import Discretization, discretize
from rfbdd.forest import Forest, Tree




def build_test_forest() -> Forest:
    t0 = Tree(
        feature=np.array([0, -1, 1, -1, -1], dtype=np.int64),
        threshold=np.array([5.7, np.nan, 10.0, np.nan, np.nan]),
        left=np.array([1, -1, 3, -1, -1], dtype=np.int64),
        right=np.array([2, -1, 4, -1, -1], dtype=np.int64),
        label=np.array([-1, 0, -1, 0, 1], dtype=np.int64),
    )

    t1 = Tree(
        feature=np.array([0, -1, 0, -1, -1], dtype=np.int64),
        threshold=np.array([2.5, np.nan, 8.1, np.nan, np.nan]),
        left=np.array([1, -1, 3, -1, -1], dtype=np.int64),
        right=np.array([2, -1, 4, -1, -1], dtype=np.int64),
        label=np.array([-1, 0, -1, 1, 0], dtype=np.int64),
    )

    return Forest(
        trees=[t0, t1],
        n_features=2,
        n_classes=2,
        classes=np.array([0, 1]),
        name="section_3_4_test",
    )


def get_test_discretization() -> Discretization:
    """Helper to get a fresh Discretization object for each test."""
    forest = build_test_forest()
    return discretize(forest)


def check(condition: bool, message: str) -> None:
    """Print PASS/FAIL and raise an assertion on failure."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    assert condition, message


# =====================================================================
# SECTION 3 TESTS
# =====================================================================

def test_thresholds() -> None:
    d = get_test_discretization()
    expected_thresholds = [np.array([2.5, 5.7, 8.1]), np.array([10.0])]
    check(np.array_equal(d.thresholds[0], expected_thresholds[0]), "Feature 0 thresholds are [2.5, 5.7, 8.1]")
    check(np.array_equal(d.thresholds[1], expected_thresholds[1]), "Feature 1 threshold is [10.0]")


def test_number_of_cells() -> None:
    d = get_test_discretization()
    check(d.n_cells == [4, 2], "Feature 0 has 4 cells and feature 1 has 2 cells")


def test_real_values_to_cell_indices() -> None:
    d = get_test_discretization()
    for value, expected in [(-1.0, 0), (2.5, 0), (4.0, 1), (5.7, 1), (7.0, 2), (8.1, 2), (9.0, 3)]:
        check(d.cell_index(0, value) == expected, f"cell_index(0, {value}) == {expected}")
    for value, expected in [(9.9, 0), (10.0, 0), (10.1, 1)]:
        check(d.cell_index(1, value) == expected, f"cell_index(1, {value}) == {expected}")


def test_sample_to_cell_indices() -> None:
    d = get_test_discretization()
    actual_cells = d.cells_of(np.array([7.0, 10.1]))
    check(np.array_equal(actual_cells, np.array([2, 1])), "cells_of([7.0, 10.1]) == [2, 1]")


def test_cell_boundaries() -> None:
    d = get_test_discretization()
    expected_bounds = [(-np.inf, 2.5), (2.5, 5.7), (5.7, 8.1), (8.1, np.inf)]
    for c, expected in enumerate(expected_bounds):
        actual = d.cell_bounds(0, c)
        check(actual[0] == expected[0] and actual[1] == expected[1], f"cell_bounds(0, {c}) is correct")


def test_tau_mapping() -> None:
    d = get_test_discretization()
    expected_taus = [{0: 1, 2: 0}, {0: 0, 2: 2}]
    check(d.taus == expected_taus, "tau mapping is correct for every internal node")


def test_threshold_equivalence() -> None:
    d = get_test_discretization()
    for value in [-1.0, 2.5, 4.0, 5.7, 7.0, 8.1, 9.0]:
        original = value <= 5.7
        cell = d.cell_index(0, value)
        discretized = cell <= d.taus[0][0]
        check(original == discretized, f"threshold equivalence holds for value {value}")


# =====================================================================
# SECTION 4 TESTS
# =====================================================================

def test_number_of_bits() -> None:
    d = get_test_discretization()
    check(d.n_bits == [2, 1], "feature 0 uses 2 bits and feature 1 uses 1 bit")


def test_bit_names() -> None:
    d = get_test_discretization()
    expected_bit_names = [["x0_0", "x0_1"], ["x1_0"]]
    check(d.bit_names == expected_bit_names, "bit names are correct")


def test_total_input_bits() -> None:
    d = get_test_discretization()
    check(d.n_input_bits == 3, "total number of input bits is 3")


def test_encode_cells() -> None:
    d = get_test_discretization()
    cases = [
        (np.array([0, 0]), {"x0_0": False, "x0_1": False, "x1_0": False}),
        (np.array([1, 0]), {"x0_0": True, "x0_1": False, "x1_0": False}),
        (np.array([2, 0]), {"x0_0": False, "x0_1": True, "x1_0": False}),
        (np.array([3, 1]), {"x0_0": True, "x0_1": True, "x1_0": True}),
    ]
    for cells, expected in cases:
        actual = d.encode_cells(cells)
        check(actual == expected, f"encode_cells({cells.tolist()}) is correct")


def test_encode_sample() -> None:
    d = get_test_discretization()
    actual = d.encode_sample(np.array([7.0, 10.1]))
    expected = {"x0_0": False, "x0_1": True, "x1_0": True}
    check(actual == expected, "encode_sample([7.0, 10.1]) is correct")


def test_decode_cells() -> None:
    d = get_test_discretization()
    cases = [
        (np.array([0, 0]), {"x0_0": False, "x0_1": False, "x1_0": False}),
        (np.array([1, 0]), {"x0_0": True, "x0_1": False, "x1_0": False}),
        (np.array([2, 0]), {"x0_0": False, "x0_1": True, "x1_0": False}),
        (np.array([3, 1]), {"x0_0": True, "x0_1": True, "x1_0": True}),
    ]
    for expected_cells, assignment in cases:
        actual = d.decode_cells(assignment)
        check(np.array_equal(actual, expected_cells), f"decode_cells({assignment}) is correct")


def test_invalid_binary_pattern() -> None:
    alias_d = Discretization(
        thresholds=[np.array([0.0, 1.0])], n_cells=[3], n_bits=[2],
        bit_names=[["x0_0", "x0_1"]], n_features=1, taus=[{}]
    )
    decoded = alias_d.decode_cells({"x0_0": True, "x0_1": True})
    check(np.array_equal(decoded, np.array([2])), "invalid binary pattern 3 aliases to the last cell 2")


def test_bit_ownership() -> None:
    d = get_test_discretization()
    expected_owner = {"x0_0": (0, 0), "x0_1": (0, 1), "x1_0": (1, 0)}
    check(d.bit_owner() == expected_owner, "every bit is associated with the correct feature and position")