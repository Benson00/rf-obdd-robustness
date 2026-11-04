# Section 6: Encoding the First Tree: the Label Counters

---

## 1. Goal

Random forest classification relies on majority voting across all $D$ trees.
The logical encoding accumulates votes class by class in binary counters.

**Section 6 establishes the base case:** encoding the contribution of tree $T_0$.
Tree $T_0$ casts the first vote, so its responsibility is to **initialize the state**:
- The counter for the class of the reached leaf is set to $1$.
- All other class counters are set to $0$.
- All win flags are initialized to $\text{False}$ (for $D \ge 2$, a single vote cannot win).

---

## 2. Theory

### Definition 5 (Winning Threshold and Label Counters)
For a forest with $D$ trees and class set $L$, a label wins as soon as it reaches:
$$W = \left\lfloor \frac{D}{2} \right\rfloor + 1$$
Because no rival can ever overturn a label with $W$ votes, counters never need to count beyond $W$.
Each label $\ell \in L$ is assigned a binary counter:
$$lc_0^\ell, lc_1^\ell, \dots, lc_M^\ell, \quad M = \left\lceil \log_2 \frac{D}{2} \right\rceil + 1$$
representing the integer:
$$LC^\ell = \sum_{j=0}^M 2^j lc_j^\ell$$

Additionally, each label $\ell$ receives a propositional flag $\text{label\_win}^\ell$, indicating whether label $\ell$ has reached certification.

For a constant integer $v$, the condition $(LC^\ell = v)$ fixes each bit:
$$(LC^\ell = v) \equiv \bigwedge_{j=0}^M (lc_j^\ell = v_j)$$
Specifically:
- $(LC^\ell = 0) \equiv \bigwedge_{j=0}^M \neg lc_j^\ell$
- $(LC^\ell = 1) \equiv lc_0^\ell \wedge \bigwedge_{j=1}^M \neg lc_j^\ell$

### Definition 6 (Path Formula)
For leaf $b$ with depth $m$, let $n_1, \dots, n_m$ be the internal nodes from the root to $b$:
$$\pi_b = \bigwedge_{i=1}^m \begin{cases} \text{nt}(f_i, \tau_i) & \text{if branch is } \le (\text{left child}) \\ \neg \text{nt}(f_i, \tau_i) & \text{if branch is } > (\text{right child}) \end{cases}$$
The path formulas across all leaves form a partition of the input space: they are pairwise mutually exclusive and their union is $\top$.

### Definition 7 (First-Tree Formula $\Phi_0$)
$$\Phi_0 = \left( \bigvee_{b \in \text{Leaves}(T_0)} \pi_b \wedge (LC^{\lambda_E(b)} = 1) \wedge \bigwedge_{\ell \neq \lambda_E(b)} (LC^\ell = 0) \right) \wedge \bigwedge_{\ell \in L} \neg \text{label\_win}^\ell$$

---

## 3. Step-by-Step Code Walkthrough (`rfbdd/first_tree.py`)

Here is how each function works in detail:

### 1. `compute_counter_width_m(d)`
Computes $M = \lceil \log_2(D / 2) \rceil + 1$.  
The counter will have $M + 1$ bits (from bit 0 to bit $M$), which is wide enough to represent any integer value from $0$ up to $W$.

### 2. `encode_counter_equals(bdd, label, val, m)`
Forces the binary counter of a specific class `label` to equal an exact integer value `val`:
- For each bit position $j \in \{0, \dots, M\}$, it reads the $j$-th bit of `val` using `(val >> j) & 1`.
- If the bit is `1`, it asserts the variable is `True`: `bdd.var(...)`.
- If the bit is `0`, it asserts the variable is `False`: `~bdd.var(...)`.
- It takes the AND (`&`) across all bits.

**Example with 2 bits ($M=1$):**
- To set $(LC^0 = 0)$: asserts `~lc_0_0 & ~lc_0_1` (both bits 0).
- To set $(LC^0 = 1)$: asserts `lc_0_0 & ~lc_0_1` (bit 0 is 1, bit 1 is 0).

### 3. `encode_path_formula(bdd, disc, steps, taus, rho=None)`
Builds the path formula $\pi_b$ that routes an input sample to leaf $b$:
- Receives `steps`, a list of tuples `(node_id, feature_idx, branch_dir)`.
- For each step:
  - Retrieves the threshold index $\tau$ for that node from `taus[node_id]`.
  - Calls `encode_node_test(bdd, disc, f, tau)` to get the boolean formula `nt`.
  - If `branch_dir == 1` (left child, test was true $\le$): keeps `nt`.
  - If `branch_dir == 0` (right child, test was false $>$): negates it `~nt`.
- Combines all steps with AND (`&`).

### 4. `encode_first_tree(bdd, forest, disc, rho=None)`
Assembles the complete $\Phi_0$ formula:
1. Iterates through all root-to-leaf paths in tree $T_0$ using `t0.paths()`.
2. For each path ending in leaf $b$ with predicted class $s = \lambda_E(b)$:
   - Computes the path condition `pi_b`.
   - Forces the winner counter $(LC^s = 1)$ using `encode_counter_equals(..., label=s, val=1)`.
   - Forces all rival counters $(LC^\ell = 0)$ for all $\ell \neq s$ using `encode_counter_equals(..., label=l, val=0)`.
   - Conjoins them: `pi_b & (LC^s == 1) & (LC^{all_others} == 0)`.
3. Takes the OR (`|`) over all leaves (since an input takes exactly one path).
4. Finally, appends the condition that no class has won yet: ANDs `~win_l` for every class $\ell$.
