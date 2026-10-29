# Section 4: Binary Encoding of the Cell Index

---

## 1. Motivation

In standard Boolean encodings of decision trees:
- Each cell $c$ receives a distinct propositional variable $C_c$.
- An **exactly-one constraint** is strictly required:
  $$(C_0 \vee C_1 \vee \dots) \wedge \bigwedge_{i \neq j} \neg (C_i \wedge C_j)$$
  This introduces $O(K^2)$ mutual exclusion clauses for $K$ cells.

**The Section 4 Alternative:**
Encode the cell index in **binary**. This eliminates the need for any constraint clauses, as every bit assignment naturally maps to a unique integer.

---

## 2. Theory

### Definition 3 (Cell Bits)
For each feature $f$ with $\text{cells}(f)$ intervals, introduce:
$$b_f = \lceil \log_2 \text{cells}(f) \rceil$$
propositional bit variables $B_f^0, B_f^1, \dots, B_f^{b_f - 1}$.

The integer cell index represented by these bits is:
$$X_f = \sum_{j=0}^{b_f - 1} 2^j B_f^j$$

### Remark 1 (No Constraint Required & Aliasing)
- Every assignment to $B_f^0, \dots, B_f^{b_f - 1}$ maps to a valid integer.
- When $\text{cells}(f)$ is not a power of 2, integers $\ge \text{cells}(f)$ are **clamped to the last cell** ($\text{cells}(f) - 1$). The last cell already represents $(l_{k-1}, +\infty)$, so all surplus bit patterns safely alias to it without corrupting the logic.

### Remark 2 (Variable Savings)
Logarithmic reduction in the number of propositional variables:
- 32 cells $\to$ 5 variables (vs 32 in one-hot).
- 128 cells $\to$ 7 variables (vs 128 in one-hot).

---

## 3. Code: `rfbdd/discretize.py`

### Properties & Metadata
- `n_bits`: List of bit counts $b_f = \lceil \log_2 \text{cells}(f) \rceil$ for each feature.
- `bit_names`: List of bit variable names (e.g., `x0_0`, `x0_1`, `x1_0`).
- `n_input_bits`: Total number of input bits across all features ($\sum b_f$).
- `all_bit_names`: Flattened list of all bit variable names in variable order.
- `bit_owner()`: Dictionary mapping each bit variable name back to `(feature_idx, bit_position)`.

### Encoding & Decoding
- `encode_cells(cells)`: Converts an array of cell indices to a boolean dictionary `{bit_name: bool}` using bitwise shift: `(cell >> j) & 1`.
- `encode_sample(x)`: Encodes a continuous sample $x \in \mathbb{R}^F$ directly into `{bit_name: bool}` via `encode_cells(cells_of(x))`.
- `decode_cells(assign)`: Reconstructs cell indices from a boolean dictionary, applying Remark 1 clamping:
  $$\text{cell}[f] = \min\left(\sum_{j} 2^j \cdot \text{assign}[B_f^j],\; \text{cells}(f) - 1\right)$$

---

## 4. Concrete Example

Suppose feature $f=0$ has 4 thresholds $[2.0, 4.0, 6.0, 8.0]$, defining **5 cells** (indices $0, 1, 2, 3, 4$):
- Cell $0$: $(-\infty, 2.0]$
- Cell $1$: $(2.0, 4.0]$
- Cell $2$: $(4.0, 6.0]$
- Cell $3$: $(6.0, 8.0]$
- Cell $4$: $(8.0, +\infty)$

Since $\lceil \log_2 5 \rceil = 3$, it uses **3 bits**: `x0_0` (weight $2^0=1$), `x0_1` (weight $2^1=2$), `x0_2` (weight $2^2=4$).

### 1. `encode_cells([3])`
- Integer 3 in binary is $011_2$:
  - `x0_0` = $(3 \gg 0) \& 1 = 1 \implies \text{True}$
  - `x0_1` = $(3 \gg 1) \& 1 = 1 \implies \text{True}$
  - `x0_2` = $(3 \gg 2) \& 1 = 0 \implies \text{False}$
- Returns `{"x0_0": True, "x0_1": True, "x0_2": False}`.
- Means: 011 that is (from right to left) -> 1 + 2 + 0 = 3
### 2. `encode_sample([5.5])`
- Continuous value $5.5 \in (4.0, 6.0] \implies \text{cell } 2$.
- Encodes cell index 2 ($010_2$) $\implies$ `{"x0_0": False, "x0_1": True, "x0_2": False}`.

### 3. `decode_cells(...)` with Clamping
- **Valid pattern**: `{"x0_0": True, "x0_1": True, "x0_2": False}` $\implies 1 + 2 + 0 = 3$. Result is $\min(3, 4) = 3$.
- **Surplus pattern (Remark 1)**: `{"x0_0": True, "x0_1": True, "x0_2": True}` $\implies 1 + 2 + 4 = 7$.
  Since max cell index is $5 - 1 = 4$, the clamping gives:
  $$\min(7, 4) = 4$$
  The surplus pattern safely aliases to the last cell $(8.0, +\infty)$.


