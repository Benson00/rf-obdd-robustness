
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from rfbdd.forest import Forest


@dataclass
class Discretization:
    """
        Stores the discretization of the feature space induced by a forest.

        The forest thresholds divide each feature axis into a finite number of
        cells. Each cell is represented by an integer index and then encoded using
        binary variables. This class stores all the information needed to move
        between real feature values, cell indices, and binary cell encodings.

        For example, if a feature has thresholds ``[2.5, 5.7, 8.1]``, it is divided
        into four cells:
        ``(-inf, 2.5]`` -> cell 0,
        ``(2.5, 5.7]`` -> cell 1,
        ``(5.7, 8.1]`` -> cell 2,
        ``(8.1, +inf)`` -> cell 3.
        These four cells require ``ceil(log2(4)) = 2`` binary variables, so cell 3
        is represented by the binary assignment ``11``.

        Attributes
        ----------
        thresholds
            Sorted distinct thresholds for each feature. These thresholds define
            the cells of each feature.
        n_cells
            Number of cells for each feature. A feature with ``k`` thresholds has
            ``k + 1`` cells.
        n_bits
            Number of binary variables needed to represent the cells of each
            feature, computed as ``ceil(log2(n_cells))``.
        bit_names
            Names of the binary variables used for each feature.
        n_features
            Total number of input features.
        taus
            Mapping from each tree node to the corresponding cell index of its
            threshold. This allows a real-valued test ``X_f <= threshold`` to be
            expressed as ``cell(X_f) <= tau``.

        Properties
        ----------
        n_input_bits
            Returns the total number of binary input variables used by all features.
        all_bit_names
            Returns a single list containing the names of all binary variables.

        Methods
        -------
        bit_owner()
            Returns the feature and bit position associated with each binary
            variable.
        cell_index(f, value)
            Returns the cell index containing a given real value of feature ``f``.
        cells_of(x)
            Converts a real-valued sample into its cell indices.
        encode_cells(cells)
            Converts cell indices into binary variable assignments.
        encode_sample(x)
            Converts a real-valued sample directly into binary variable
            assignments.
        decode_cells(assign)
            Converts binary variable assignments back into cell indices.
        cell_bounds(f, c)
            Returns the real interval corresponding to cell ``c``.
        cell_extents(f, clip)
            Returns a finite width for each cell of a feature.
        bit_weights(data_ranges)
            Computes the weights of the binary variables used for weighted model
            counting.
    """

    thresholds: list[np.ndarray]      # sorted distinct thresholds per feature
    n_cells: list[int]                # cells(f) = len(thresholds[f]) + 1
    n_bits: list[int]                 # b_f = ceil(log2 cells(f))
    bit_names: list[list[str]]        # bit_names[f][j] -> name of B^j_f
    n_features: int
    # tau of every internal node of every tree: taus[t][node] = cell index of
    # the node threshold, so the test "X_f <= threshold" becomes "X_f <= tau".
    taus: list[dict[int, int]]

    # ------------------------------------------------------------------ #
    @property
    def n_input_bits(self) -> int:
        """Return the total number of binary input variables"""
        return sum(self.n_bits)

    @property
    def all_bit_names(self) -> list[str]:
        """Return the names of all binary input variables"""
        return [nm for names in self.bit_names for nm in names]

    def bit_owner(self) -> dict[str, tuple[int, int]]:
        """
        Return the feature and bit position associated with each input bit.

        Example: If ``x2_1`` is the second bit of feature 2, the mapping contains: {"x2_1": (2, 1)}
        """
        return {nm: (f, j) for f, names in enumerate(self.bit_names) for j, nm in enumerate(names)}

    # ------------------------------------------------------------------ #
    def cell_index(self, f: int, value: float) -> int:
        """
        Return Cell of ``value`` on axis ``f``. \\
        Given a real value of feature f, return the index of the cell it belongs to. \\
        The cell index is determined by the number of thresholds that are less than or equal to the value.          
        """
        return bisect.bisect_left(self.thresholds[f], value)

    def cells_of(self, x: np.ndarray) -> np.ndarray:
        """The tuple of cell indices of a real sample. Works on an entire sample at once."""
        return np.array([self.cell_index(f, x[f]) for f in range(self.n_features)], dtype=np.int64)

    def encode_cells(self, cells: np.ndarray) -> dict[str, bool]:
        """Cell indices -> assignment of the cell bits ``B^j_f``. \\
        Given an array of cell indices for each feature, return a dictionary mapping each bit name to its corresponding boolean value. \\    
        """
        out: dict[str, bool] = {}
        for f in range(self.n_features):
            c = int(cells[f])
            for j, nm in enumerate(self.bit_names[f]):
                out[nm] = bool((c >> j) & 1)
        return out

    def encode_sample(self, x: np.ndarray) -> dict[str, bool]:
        """Encode all the sample."""
        return self.encode_cells(self.cells_of(x))

    def decode_cells(self, assign: dict[str, bool]) -> np.ndarray:
        """From assignments of the cell bits ``B^j_f`` to the cell indices."""
        cells = np.zeros(self.n_features, dtype=np.int64)
        for f in range(self.n_features):
            v = 0
            for j, nm in enumerate(self.bit_names[f]):
                if assign.get(nm, False):
                    v |= 1 << j
            cells[f] = min(v, self.n_cells[f] - 1)
        return cells

    # ------------------------------------------------------------------ #
    def cell_bounds(self, f: int, c: int) -> tuple[float, float]:
        """Real interval of cell ``c`` on axis ``f`` (``-inf`` / ``+inf`` ends)."""
        th = self.thresholds[f]
        lo = -math.inf if c == 0 else float(th[c - 1])
        hi = math.inf if c >= len(th) else float(th[c])
        return lo, hi

    def cell_extents(self, f: int, clip: tuple[float, float] | None = None) -> np.ndarray:
        """
        Real-space width of every cell of feature ``f``.

        The two unbounded end cells are clipped to ``clip`` (by default the
        observed data range widened by one median interior width) so that the
        measure is finite.  Every pattern that aliases onto the
        last cell is handled by :meth:`bit_weights`.
        """
        th = np.asarray(self.thresholds[f], dtype=float)
        k = len(th)
        if k == 0:
            return np.array([1.0])
        widths = np.diff(th) if k > 1 else np.array([], dtype=float)
        pad = float(np.median(widths)) if widths.size else max(1.0, abs(float(th[0])) * 0.1 + 1e-9)
        lo = clip[0] if clip is not None else float(th[0]) - pad
        hi = clip[1] if clip is not None else float(th[-1]) + pad
        edges = np.concatenate(([min(lo, float(th[0]) - pad)], th, [max(hi, float(th[-1]) + pad)]))
        ext = np.diff(edges)
        return np.maximum(ext, 1e-12)

    def bit_weights(self, data_ranges: list[tuple[float, float]] | None = None) -> dict[str, float]:
        """Per-bit weights ``w_k`` for the weighted count of Proposition 5.

        Proposition 5 scales a branch setting ``x_k`` by ``w_k`` and a branch
        clearing it by ``1 - w_k``, so the weighted count is the measure of the
        *product* distribution with those marginals.  We choose ``w_k`` as the
        marginal probability that bit ``k`` is 1 under the extent-induced
        distribution over the ``2^{b_f}`` bit patterns of its feature, with the
        aliased patterns folded onto the last cell (Remark 8) and each feature
        normalised to total mass 1.  The result is the mean-field product
        measure expressible in Proposition 5's per-variable form; it is *exact*
        whenever the extent distribution factorises over the bits (in
        particular for the uniform / power-of-two case, where it reduces to
        ``w_k = 1/2`` and Proposition 5 collapses to Theorem 1).
        """
        w: dict[str, float] = {}
        for f in range(self.n_features):
            b = self.n_bits[f]
            if b == 0:
                continue
            ext = self.cell_extents(f, None if data_ranges is None else data_ranges[f])
            last = self.n_cells[f] - 1
            # mass of every bit pattern in 0..2^b-1, aliases folded onto `last`
            mass = np.zeros(1 << b)
            n_alias = (1 << b) - self.n_cells[f] + 1     # patterns mapping to `last`
            for p in range(1 << b):
                c = min(p, last)
                mass[p] = ext[c] / (n_alias if c == last else 1)
            mass /= mass.sum()
            for j, nm in enumerate(self.bit_names[f]):
                sel = np.array([(p >> j) & 1 for p in range(1 << b)], dtype=bool)
                w[nm] = float(mass[sel].sum())
        return w


def discretize(forest: Forest) -> Discretization:
    """Build the discretization induced by ``forest``."""
    
    per_feature: list[set[float]] = [set() for _ in range(forest.n_features)]
    # for each feature, collect the thresholds tested on it across all trees
    for tree in forest.trees:
        for i in tree.internal_nodes():
            per_feature[int(tree.feature[i])].add(float(tree.threshold[i]))
    # sort them to define the cells of that feature.
    thresholds = [np.array(sorted(s), dtype=np.float64) for s in per_feature]
    # compute the number of cells and bits per feature, and the bit names
    n_cells = [len(t) + 1 for t in thresholds]
    # b_f = ceil(log2 cells(f)); a feature never tested has cells(f) = 1 and so
    # zero bits -- it is invisible to the forest and carries no variable.
    n_bits = [max(0, math.ceil(math.log2(c))) if c > 1 else 0 for c in n_cells]
    bit_names = [[f"x{f}_{j}" for j in range(n_bits[f])] for f in range(forest.n_features)]

    taus: list[dict[int, int]] = []
    for tree in forest.trees:
        d: dict[int, int] = {}
        for i in tree.internal_nodes():
            f = int(tree.feature[i])
            # index of this threshold in the sorted list == cell index tau such
            # that (X_f <= threshold)  <=>  (cell(X_f) <= tau)
            d[i] = int(np.searchsorted(thresholds[f], float(tree.threshold[i])))
        taus.append(d)

    return Discretization(
        thresholds=thresholds,
        n_cells=n_cells,
        n_bits=n_bits,
        bit_names=bit_names,
        n_features=forest.n_features,
        taus=taus,
    )