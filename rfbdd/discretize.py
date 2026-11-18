from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
import numpy as np

from rfbdd.forest import Forest


@dataclass
class Discretization:
    """Feature-space discretization and binary cell encoding."""

    thresholds: list[np.ndarray]      # sorted distinct thresholds per feature
    n_cells: list[int]                # cells(f) = len(thresholds[f]) + 1
    n_bits: list[int]                 # b_f = ceil(log2 cells(f))
    bit_names: list[list[str]]        # bit variable names per feature
    n_features: int                   # total number of features F
    taus: list[dict[int, int]]        # taus[t][node] = get the tau (cell index of the threshold) given an internal node of a tree t.

    def cells(self, f: int) -> int:
        """Return cells(f)."""
        return self.n_cells[f]

    def cell_index(self, f: int, value: float) -> int:
        """
        Take a real value and return the index of the cell it belongs to for feature f. It performs a binary search on the sorted thresholds for feature f to find the appropriate cell index.
        """
        return bisect.bisect_left(self.thresholds[f], float(value))

    def cells_of(self, x: np.ndarray) -> np.ndarray:
        """
        Performs mapping: R^F -> N^F.
        
        Maps a real sample x into its corresponding cell indices for each feature. It applies the cell_index method for each feature in the sample x and returns an array of cell indices.
        """
        x = np.asarray(x, dtype=float)
        return np.array([self.cell_index(f, x[f]) for f in range(self.n_features)], dtype=np.int64)

    def cell_bounds(self, f: int, c: int) -> tuple[float, float]:
        """
        Given a cell index c for feature f, return the lower and upper bounds of the cell. The lower bound is -infinity if c is 0, otherwise it is the threshold at index c-1. The upper bound is +infinity if c is greater than or equal to the number of thresholds, otherwise it is the threshold at index c.
        """
        if c < 0 or c >= self.n_cells[f]:
            raise ValueError(f"Cell index {c} out of range for feature {f} (cells={self.n_cells[f]})")
        th = self.thresholds[f]
        lo = -math.inf if c == 0 else float(th[c - 1])
        hi = math.inf if c >= len(th) else float(th[c])
        return lo, hi

    @property
    def n_input_bits(self) -> int:
        """
        Return the total number of binary input variables across all features.
        """
        return sum(self.n_bits)

    @property
    def all_bit_names(self) -> list[str]:
        """
        Return a flat list of all binary variable names in order.
        """
        return [nm for names in self.bit_names for nm in names]

    def bit_owner(self) -> dict[str, tuple[int, int]]:
        """
        Return a dictionary mapping each bit variable name to its owner (feature index, bit position).

        e.g., if bit_names[0] = ["x0_0", "x0_1"], then bit_owner() will return {"x0_0": (0, 0), "x0_1": (0, 1)}.
        """
        return {nm: (f, j) for f, names in enumerate(self.bit_names) for j, nm in enumerate(names)}

    def encode_cells(self, cells: np.ndarray) -> dict[str, bool]:
        """
        Convert an array of cell indices to a boolean assignment dictionary.
        
        For each feature f, it takes the cell index c and converts it to its binary representation. Each bit in the binary representation is assigned to the corresponding bit variable name for that feature. The result is a dictionary mapping each bit variable name to its boolean value (True or False).

        e.g. consider [2,1] for 2 features with 3 and 2 cells respectively. The binary representation of 2 is '10' and for 1 is '01'. The resulting dictionary will be {"x0_0": False, "x0_1": True, "x1_0": True}.
        """
        out: dict[str, bool] = {}
        for f in range(self.n_features):
            c = int(cells[f])
            for j, nm in enumerate(self.bit_names[f]):
                out[nm] = bool((c >> j) & 1) # shift the cell index c right by j bits and check if the least significant bit is 1 (True) or 0 (False). e.g. for c=2 (binary '10'), j=0 gives False, j=1 gives True.
        return out

    def encode_sample(self, x: np.ndarray) -> dict[str, bool]:
        """
        Convert a real-valued sample vector x to a boolean bit assignment.
        It first maps the real-valued sample x to its corresponding cell indices using the cells_of method, and then encodes those cell indices into a boolean assignment dictionary using the encode_cells method.
        """
        return self.encode_cells(self.cells_of(x))

    def decode_cells(self, assign: dict[str, bool]) -> np.ndarray:
        """
        Decode a boolean assignment back to cell indices with clamping.

        e.g. input: {"x0_0": False, "x0_1": True, "x1_0": True} for 2 features with 3 and 2 cells respectively. The binary representation for feature 0 is '10' (which is 2) and for feature 1 is '01' (which is 1). The resulting cell indices will be [2, 1]. In case the decoded cell index exceeds the maximum number of cells for a feature, it is clamped to the maximum valid index (n_cells[f] - 1). For example, if the decoded cell index for feature 0 was 3 (which exceeds the maximum of 2), it would be clamped to 2.
        """
        cells = np.zeros(self.n_features, dtype=np.int64)
        for f in range(self.n_features):
            v = 0
            for j, nm in enumerate(self.bit_names[f]):
                if assign.get(nm, False):
                    v |= (1 << j)
            cells[f] = min(v, self.n_cells[f] - 1)
        return cells


def discretize(forest: Forest) -> Discretization:
    """Build the discretization and binary encoding induced by forest."""

    # Collect thresholds per feature across all trees
    per_feature: list[set[float]] = [set() for _ in range(forest.n_features)]

    # collect thresholds in the forest
    for tree in forest.trees:   # for all trees in the forest
        for node in tree.internal_nodes():  # for all internal nodes in the tree
            f = int(tree.feature[node]) # extract the feature index for the node
            per_feature[f].add(float(tree.threshold[node])) # add the threshold to the set for that feature

    # sort thresholds
    thresholds = [np.array(sorted(s), dtype=np.float64) for s in per_feature]
    # compute number of cells -> cells(f)=k+1
    n_cells = [len(t) + 1 for t in thresholds]
    # compute number of bits needed for each feature -> b_f = ceil(log2 cells(f))
    n_bits = [max(0, math.ceil(math.log2(c))) if c > 1 else 0 for c in n_cells]
    # assign names to each bit variable for each feature (e.g. "x0_0", "x0_1" for a feature with 2 bits)
    bit_names = [[f"x{f}_{j}" for j in range(n_bits[f])] for f in range(forest.n_features)]


    # define taus as a list that contains a dictionary for each tree in the forest, that maps each internal node of the tree to its corresponding threshold cell index (e.g. tau[node] = cell index for the threshold at that node)
    taus: list[dict[int, int]] = []
    # for each tree in the forest
    for tree in forest.trees:
        # create a dictionary to hold the mapping of internal nodes to their corresponding threshold cell indices for the current tree
        tree_taus: dict[int, int] = {}
        # for each internal node in the current tree
        for node in tree.internal_nodes():
            # extract the feature index for the current internal node
            f = int(tree.feature[node])
            # find the index of the threshold for the current node in the sorted list of thresholds for that feature
            tree_taus[node] = int(np.searchsorted(thresholds[f], float(tree.threshold[node])))
        # append the mapping for the current tree
        taus.append(tree_taus)

    return Discretization(
        thresholds=thresholds,
        n_cells=n_cells,
        n_bits=n_bits,
        bit_names=bit_names,
        n_features=forest.n_features,
        taus=taus,
    )