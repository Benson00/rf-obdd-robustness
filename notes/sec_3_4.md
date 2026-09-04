# Sections 3 and 4 — Discretization and Binary Encoding

This document maps the assignment Sections 3 and 4 to the implementation in
`rfbdd/discretize.py` and explains what each relevant piece of code does.

---

## Section 3 — The Discretization Induced by the Forest

### Goal

A trained random forest uses threshold tests such as:

```text
X_f <= 5.7
```

All thresholds used on the same feature divide its real axis into cells.
The discretization is deterministic: no probability or learning is involved.

For example, if a feature has thresholds:

```text
[2.5, 5.7, 8.1]
```

the cells are:

```text
cell 0 = (-inf, 2.5]
cell 1 = (2.5, 5.7]
cell 2 = (5.7, 8.1]
cell 3 = (8.1, +inf)
```

A real value is represented by the index of the cell containing it.

### Code location

The implementation is in:

```text
rfbdd/discretize.py
```

The main function is:

```python
discretize(forest)
```

and the resulting information is stored in:

```python
Discretization
```

### Definition 1 — Cells of a feature

The code first collects all thresholds used by all internal nodes of all
trees, separately for each feature:

```python
per_feature = [set() for _ in range(forest.n_features)]

for tree in forest.trees:
    for i in tree.internal_nodes():
        per_feature[int(tree.feature[i])].add(float(tree.threshold[i]))
```

Then the thresholds are sorted:

```python
thresholds = [
    np.array(sorted(s), dtype=np.float64)
    for s in per_feature
]
```

This implements the idea:

```text
collect thresholds -> remove duplicates -> sort
```

The resulting `thresholds[f]` contains the ordered thresholds for feature `f`.

### Definition 2 — Number of cells

A feature with `k` distinct thresholds has `k + 1` cells:

```python
n_cells = [len(t) + 1 for t in thresholds]
```

Therefore:

```text
cells(f) = len(thresholds[f]) + 1
```

### Mapping a real value to a cell

The method:

```python
cell_index(f, value)
```
return the cell index.

For thresholds `[2.5, 5.7, 8.1]`:

```text
2.0 -> cell 0
2.5 -> cell 0
4.0 -> cell 1
5.7 -> cell 1
7.0 -> cell 2
9.0 -> cell 3
```

The boundary convention is consistent with the assignment's
left-open/right-closed cells.

### Discretizing a complete sample

The method:

```python
cells_of(x)
```

applies `cell_index` to every feature:

```text
real sample
    |
    v
[cell(x_0), cell(x_1), ..., cell(x_F-1)]
```

For example:

```text
x = [4.0, 7.0]
```

could become:

```text
cells = [1, 2]
```

This is the implementation of the sample-to-cell mapping described by
Proposition 1.

### Cell boundaries

The method:

```python
cell_bounds(f, c)
```

returns the real interval corresponding to a cell.

For example:

```text
thresholds = [2.5, 5.7, 8.1]

cell_bounds(0, 0) -> (-inf, 2.5]
cell_bounds(0, 1) -> (2.5, 5.7]
cell_bounds(0, 2) -> (5.7, 8.1]
cell_bounds(0, 3) -> (8.1, +inf)
```

This is a helper for working explicitly with the cells defined in Section 3.

### `taus`

The `taus` structure is also computed in `discretize(forest)`:

```python
d[i] = int(
    np.searchsorted(
        thresholds[f],
        float(tree.threshold[i])
    )
)
```

For a tree node whose original test is:

```text
X_f <= 5.7
```

and where `5.7` corresponds to cell boundary index `1`, the code stores:

```text
tau = 1
```

so that later the real test can be expressed using the discretized value as:

```text
cell(X_f) <= tau
```

This is mainly the bridge from Section 3 to the later OBDD encoding.

---

# Section 4 — Binary Encoding of the Cell Index

### Goal

Section 4 replaces one Boolean variable per cell with a binary representation
of the cell index.

If a feature has `K` cells, the number of bits is:

```text
b_f = ceil(log2(K))
```

For example:

```text
K = 4
b_f = ceil(log2(4)) = 2
```

so the four cells can be represented as:

```text
cell 0 -> 00
cell 1 -> 01
cell 2 -> 10
cell 3 -> 11
```

### Code location

The implementation is also in:

```text
rfbdd/discretize.py
```

and is created by:

```python
discretize(forest)
```

### Definition 3 — Number of cell bits

The number of bits is computed with:

```python
n_bits = [
    max(0, math.ceil(math.log2(c))) if c > 1 else 0
    for c in n_cells
]
```

This implements:

```text
b_f = ceil(log2(cells(f)))
```

A feature that is never tested has one cell and therefore needs zero bits.

### Binary variable names

The code creates the Boolean variable names with:

```python
bit_names = [
    [f"x{f}_{j}" for j in range(n_bits[f])]
    for f in range(forest.n_features)
]
```

For example, if feature `2` needs three bits:

```text
x2_0
x2_1
x2_2
```

These correspond to the bits of the binary cell index.

### Encoding a cell index

The method:

```python
encode_cells(cells)
```

converts integer cell indices into Boolean assignments.

For example:

```text
cell = 3
binary = 11
```

therefore:

```text
bit 0 = True
bit 1 = True
```

The relevant operation is:

```python
bool((c >> j) & 1)
```

which extracts bit `j` from the integer cell index.

### Encoding a complete real sample

The method:

```python
encode_sample(x)
```

combines Sections 3 and 4:

```text
real sample
    |
    v
cells_of(x)
    |
    v
cell indices
    |
    v
encode_cells(...)
    |
    v
Boolean bit assignment
```

Therefore it is the direct implementation of:

```text
R^F -> N^F -> {0,1}^B
```

### Decoding

The method:

```python
decode_cells(assign)
```

performs the reverse operation:

```text
Boolean bits
    |
    v
integer cell indices
```

The expression:

```python
v |= 1 << j
```

reconstructs the integer represented by the selected bits.

If there are unused binary patterns because the number of cells is not a
power of two, the code maps those patterns to the last valid cell:

```python
cells[f] = min(v, self.n_cells[f] - 1)
```

This implements the convention described in Remark 1.

### Variable saving

The property:

```python
n_input_bits
```

returns:

```python
sum(self.n_bits)
```

so it gives the total number of Boolean input variables.

For example:

```text
32 cells:
one-hot encoding -> 32 variables
binary encoding  -> 5 variables
```

because:

```text
ceil(log2(32)) = 5
```

---

# Summary of the code mapping

| Assignment concept | Code | Section |
|---|---|---|
| Collect thresholds from all trees | `per_feature` loop | 3 |
| Sort thresholds | `thresholds` | 3 |
| Number of cells | `n_cells` | 3 |
| Real value -> cell | `cell_index()` | 3 |
| Real sample -> cells | `cells_of()` | 3 |
| Cell -> real interval | `cell_bounds()` | 3 |
| Threshold -> cell threshold | `taus` | 3 / bridge |
| Number of binary bits | `n_bits` | 4 |
| Binary variable names | `bit_names` | 4 |
| Cell -> binary assignment | `encode_cells()` | 4 |
| Real sample -> binary assignment | `encode_sample()` | 3 + 4 |
| Binary assignment -> cell | `decode_cells()` | 4 |
| Total number of input bits | `n_input_bits` | 4 |
| Flatten all bit names | `all_bit_names` | 4 |
| Bit -> feature/position | `bit_owner()` | 4 |

## Important scope

For Sections 3 and 4, the central pipeline is:

```text
Forest thresholds
      |
      v
   cells
      |
      v
cell indices
      |
      v
 binary bits
```

The methods `cell_extents()` and `bit_weights()` are not part of the core
implementation of Sections 3 and 4. They are used later for real-space
measure and weighted model counting.
