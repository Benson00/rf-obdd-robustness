from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
import numpy as np

from rfbdd.forest import Forest


@dataclass
class Discretization:
    """Feature-space discretization and binary cell encoding (Sections 3 & 4)."""

    thresholds: list[np.ndarray]      # sorted distinct thresholds per feature
    n_cells: list[int]                # cells(f) = len(thresholds[f]) + 1
    n_bits: list[int]                 # b_f = ceil(log2 cells(f))
    bit_names: list[list[str]]        # bit variable names per feature
    n_features: int                   # total number of features F
    taus: list[dict[int, int]]        # taus[t][node] = threshold cell index tau

    def cells(self, f: int) -> int:
        """Return cells(f), the number of cells for feature f (Definition 2)."""
        return self.n_cells[f]

    def cell_index(self, f: int, value: float) -> int:
        """Return the cell index in {0, ..., cells(f) - 1} for a real value on feature f."""
        return bisect.bisect_left(self.thresholds[f], float(value))

    def cells_of(self, x: np.ndarray) -> np.ndarray:
        """Convert a real-valued sample x into its tuple of cell indices (Proposition 1)."""
        x = np.asarray(x, dtype=float)
        return np.array([self.cell_index(f, x[f]) for f in range(self.n_features)], dtype=np.int64)

    def cell_bounds(self, f: int, c: int) -> tuple[float, float]:
        """Return the real interval (lower, upper) of cell c for feature f."""
        if c < 0 or c >= self.n_cells[f]:
            raise ValueError(f"Cell index {c} out of range for feature {f} (cells={self.n_cells[f]})")
        th = self.thresholds[f]
        lo = -math.inf if c == 0 else float(th[c - 1])
        hi = math.inf if c >= len(th) else float(th[c])
        return lo, hi

    @property
    def n_input_bits(self) -> int:
        """Total number of binary input variables across all features."""
        return sum(self.n_bits)

    @property
    def all_bit_names(self) -> list[str]:
        """Flat list of all binary variable names in order."""
        return [nm for names in self.bit_names for nm in names]

    def bit_owner(self) -> dict[str, tuple[int, int]]:
        """Map each bit variable name to (feature_index, bit_position)."""
        return {nm: (f, j) for f, names in enumerate(self.bit_names) for j, nm in enumerate(names)}

    def encode_cells(self, cells: np.ndarray) -> dict[str, bool]:
        """Convert an array of cell indices to a boolean assignment dictionary."""
        out: dict[str, bool] = {}
        for f in range(self.n_features):
            c = int(cells[f])
            for j, nm in enumerate(self.bit_names[f]):
                out[nm] = bool((c >> j) & 1)
        return out

    def encode_sample(self, x: np.ndarray) -> dict[str, bool]:
        """Convert a real-valued sample vector x to a boolean bit assignment."""
        return self.encode_cells(self.cells_of(x))

    def decode_cells(self, assign: dict[str, bool]) -> np.ndarray:
        """Decode a boolean assignment back to cell indices with clamping."""
        cells = np.zeros(self.n_features, dtype=np.int64)
        for f in range(self.n_features):
            v = 0
            for j, nm in enumerate(self.bit_names[f]):
                if assign.get(nm, False):
                    v |= (1 << j)
            cells[f] = min(v, self.n_cells[f] - 1)
        return cells


def discretize(forest: Forest) -> Discretization:
    """Build the discretization and binary encoding induced by forest (Sections 3 & 4)."""
    per_feature: list[set[float]] = [set() for _ in range(forest.n_features)]

    for tree in forest.trees:
        for node in tree.internal_nodes():
            f = int(tree.feature[node])
            per_feature[f].add(float(tree.threshold[node]))

    thresholds = [np.array(sorted(s), dtype=np.float64) for s in per_feature]
    n_cells = [len(t) + 1 for t in thresholds]
    n_bits = [max(0, math.ceil(math.log2(c))) if c > 1 else 0 for c in n_cells]
    bit_names = [[f"x{f}_{j}" for j in range(n_bits[f])] for f in range(forest.n_features)]

    taus: list[dict[int, int]] = []
    for tree in forest.trees:
        tree_taus: dict[int, int] = {}
        for node in tree.internal_nodes():
            f = int(tree.feature[node])
            tree_taus[node] = int(np.searchsorted(thresholds[f], float(tree.threshold[node])))
        taus.append(tree_taus)

    return Discretization(
        thresholds=thresholds,
        n_cells=n_cells,
        n_bits=n_bits,
        bit_names=bit_names,
        n_features=forest.n_features,
        taus=taus,
    )
