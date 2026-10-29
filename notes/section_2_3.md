# Section 2 and 3: Forest Representation and Discretization

---

## Section 2: Setting and Notation

### Theory
A random forest is a finite collection of decision trees:
$$\mathcal{F} = \{T_0, T_1, \dots, T_{D-1}\}$$

- **Internal nodes ($\nu$)**: Each internal node performs a single-feature threshold test of the form:
  $$x_f \le \text{threshold}$$
  - Branching: Left child is the **true** branch ($\le$), right child is the **false** branch ($>$).
- **Leaf nodes ($\lambda_E$)**: Each leaf assigns a class label from the set of classes $L$.
- **Majority Voting**: Each tree casts one vote for the class of its reached leaf. The winning threshold is:
  $$W = \lfloor D / 2 \rfloor + 1$$

---

### Code: `rfbdd/forest.py`

#### `Tree`
Stores a tree compactly using parallel 1D NumPy arrays (matching scikit-learn's internal layout):
- `feature[i]`: Feature index tested at node `i` (`-1` for leaves).
- `threshold[i]`: Split threshold (`NaN` for leaves).
- `left[i]`, `right[i]`: Indices of left/right children (`< 0` indicates a leaf).
- `label[i]`: Class label at leaf `i` (`-1` for internal nodes).

**Key methods:**
- `is_leaf(node)`: Checks if `left[node] < 0`.
- `decide(x)`: Traverses the tree for a real-valued input vector `x` using `<=` tests and returns the leaf class label.
- `decide_cells(cells, taus)`: Traverses the tree directly on discrete cell indices using `cells[f] <= tau`.
- `paths()`: Returns all root-to-leaf paths as tuples of `(node, feature, branch_direction)` along with the leaf label.

#### `Forest`
Encapsulates the collection of trees:
- `trees`: List of `Tree` instances.
- `D`: Number of trees in the forest.
- `W`: Majority threshold $\lfloor D / 2 \rfloor + 1$.
- `prefix(k)`: Returns a sub-forest containing only the first $k$ trees (recomputing $D$ and $W$), needed for incremental scaling experiments.
- `votes(x)` / `predict_majority(X)`: Aggregates tree predictions via majority voting.
- `load_forest(path)`: Loads a serialized scikit-learn model (`.pkl`) and converts it to our `Forest` structure.

---

## Section 3: The Discretization Induced by the Forest

### Theory
Because all tree splits are axis-aligned conditions of the form $x_f \le \text{threshold}$, the forest deterministically discretizes the continuous input space $\mathbb{R}^F$.

#### Definition 1 (Cells of a Feature)
For each feature $f$, collect every threshold tested across all trees and sort the distinct values:
$$\ell_0 < \ell_1 < \dots < \ell_{k-1}$$

These $k$ thresholds partition the real line into $k + 1$ intervals called **cells**:
$$(-\infty, \ell_0],\; (\ell_0, \ell_1],\; \dots,\; (\ell_{k-2}, \ell_{k-1}],\; (\ell_{k-1}, +\infty)$$

> **Interval convention**: Left-open and right-closed $(a, b]$ because tests are $\le$. A value equal to the threshold falls into the lower cell.

#### Definition 2 (Cell-Count Function)
$$\text{cells}(f) = k + 1$$
Cell indices range over $\{0, 1, \dots, \text{cells}(f) - 1\}$.

#### Proposition 1 (Lossless Discretization)
Mapping real samples to cell index tuples:
$$\mathbb{R}^F \longrightarrow \mathbb{N}^F$$
is completely **lossless** for the forest:
- Any two inputs in the same cell take identical branches at every node.
- A test $x_f \le \ell_\tau$ is exactly equivalent to:
  $$\text{cell}(x_f) \le \tau$$
  where $\tau \in \{0, \dots, \text{cells}(f) - 2\}$ is the rank index of threshold $\ell_\tau$.

---

### Code: `rfbdd/discretize.py`

#### `Discretization`
Data class containing the discretization parameters:
- `thresholds`: Sorted unique thresholds for each feature.
- `n_cells`: List where `n_cells[f] = len(thresholds[f]) + 1`.
- `taus`: List of dictionaries (one per tree) mapping `node_id -> cell_index_tau`.

**Key methods:**
- `cell_index(f, value)`: Uses binary search (`bisect_left`) to return the cell index for a scalar value on feature $f$.
- `cells_of(x)`: Converts a continuous sample vector $x \in \mathbb{R}^F$ into an integer cell vector in $\mathbb{N}^F$.
- `cell_bounds(f, c)`: Returns the real interval $(-\infty, \ell_0], \dots$ for cell $c$ of feature $f$.

#### `discretize(forest)`
Constructs the `Discretization` object from a `Forest`:
1. Gathers all unique thresholds tested on each feature across all trees.
2. Sorts them in ascending order.
3. Computes $\tau$ for each internal node using `np.searchsorted`.

