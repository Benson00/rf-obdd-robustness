# Section 7: Encoding a Subsequent Tree: the Inductive Step

---

## 1. Goal

While tree $T_0$ initialized the counters, every later tree:
$$T_t \quad (1 \le t \le D - 1)$$
reads the running tally from stage $(t-1)$, adds its vote, and updates the win flags.

Each subsequent tree performs arithmetic:
- It carries each counter forward unchanged unless its leaf selects that label, in which case it adds 1.
- If the vote clinches the win for a label, it moves directly to certifying the victory and shuts down further counting.
- If a winner was already decided earlier, the tree is **deactivated** ($\text{Hold}$): it copies the verdict and frees the counter bits.

---

## 2. Theory

### Definition 8 (Staged State and $R_t$)
To chain the trees, each tree operates on two copies of the state:
- **Incoming stage-$(t-1)$ state**: unbarred variables $LC^\ell$ (`lc_l_j`) and $\text{label\_win}^\ell$ (`win_l`).
- **Outgoing stage-$t$ state**: barred variables $\overline{LC}^\ell$ (`lc_bar_l_j`) and $\overline{\text{label\_win}}^\ell$ (`win_bar_l`).

The number of trees still remaining to vote after $T_t$ is the constant:
$$R_t = D - 1 - t$$

### Definition 9 (Clinching Predicate $\text{Clinch}_s^{(t)}$)
Tree $T_t$ votes for label $s = \lambda_E(b)$. This vote makes $s$ the certain winner when no rival $\ell \neq s$ could overtake $s$ even if every one of the $R_t$ remaining trees voted for that rival:
$$\text{Clinch}_s^{(t)} \equiv \bigwedge_{\ell \neq s} \left( LC^s \ge LC^\ell + R_t \right)$$

#### Earliest-Certification Short-Circuit
Because counters never exceed $W - 1$ in practice, whenever:
$$R_t > W - 1 \quad \left(\text{equivalently } t < \lfloor D / 2 \rfloor\right)$$
$\text{Clinch}_s^{(t)}$ is identically **$\bot$ (False)**. No comparator circuit needs to be built!

#### Chain-Free Offset Comparator ($\chi_M$)
When $t \ge \lfloor D / 2 \rfloor$, each condition $LC^s \ge LC^\ell + R_t$ is encoded without auxiliary variables using independent carries:
$$\kappa_j \equiv \left( (b_{j-1}\dots b_0)_2 \ge 2^j - (R_t \bmod 2^j) \right), \quad \kappa_0 = \bot$$
$$\hat{g}_j = b_j \oplus r_j \oplus \kappa_j$$
$$\chi_{-1} = \top, \quad \chi_j = (a_j \wedge \neg \hat{g}_j) \vee \big( (a_j \leftrightarrow \hat{g}_j) \wedge \chi_{j-1} \big)$$
The top formula $\chi_M$ encodes $LC^s \ge LC^\ell + R_t$.

### Definition 10 (Subsequent-Tree Formula $\Phi_t$)
Let $\text{won} \equiv \bigvee_{\ell \in L} \text{label\_win}^\ell$. The tree formula is a guarded choice:
$$\Phi_t = (\text{won} \wedge \text{Hold}) \vee (\neg \text{won} \wedge \text{Step}_t)$$

1. **Hold Branch**:
   $$\text{Hold} \equiv \bigwedge_{\ell \in L} (\overline{\text{label\_win}}^\ell \leftrightarrow \text{label\_win}^\ell)$$
   Leaves all counters $\overline{LC}^\ell$ unconstrained (don't-cares), drastically simplifying the OBDD.

2. **Step Branch**:
   $$\text{Step}_t \equiv \bigvee_{b \in \text{Leaves}(T_t)} \pi_b \wedge \Big[ (\text{Clinch}_s^{(t)} \wedge \text{Win}_s) \vee (\neg \text{Clinch}_s^{(t)} \wedge \text{Inc}_s) \Big]$$
   where $s = \lambda_E(b)$:
   - **$\text{Win}_s$**: $\overline{\text{label\_win}}^s = \top$, all other win flags $\bot$, and counters freed.
   - **$\text{Inc}_s$**: $\overline{LC}^s = LC^s + 1$, all rival counters copied $\overline{LC}^\ell = LC^\ell$, and all win flags $\bot$.

---

## 3. Step-by-Step Code Walkthrough (`rfbdd/subsequent_tree.py`)

### 1. `encode_geq_offset(bdd, label_s, label_l, r_t, m)`
Implements the Definition 9 chain-free comparator:
- Precomputes carries $\kappa_j$ for $j \in \{1, \dots, M\}$ using the low $j$ bits of $LC^\ell$ against threshold $2^j - (R_t \bmod 2^j)$.
- Computes sum bits $\hat{g}_j = b_j \oplus r_j \oplus \kappa_j$.
- Evaluates the bitwise comparison recurrence $\chi_j$:
  - `strictly_greater = a_j & ~g_hat_j`
  - `equal = (a_j <-> g_hat_j)`
  - `chi = strictly_greater | (equal & chi_prev)`

### 2. `encode_clinch(bdd, label_s, t_idx, forest, m)`
Checks if label $s$ clinches at step $t$:
- Evaluates $R_t = D - 1 - t$.
- If $R_t > W - 1$, returns `bdd.false` immediately (syntactic short-circuit).
- Otherwise, takes the AND of `encode_geq_offset` over all rival classes $\ell \neq s$.

### 3. `encode_counter_copy(bdd, label, m)` and `encode_counter_increment(bdd, label, m)`
- `encode_counter_copy`: asserts each bit `lc_bar_l_j <-> lc_l_j`.
- `encode_counter_increment`: performs binary addition $+1$ using a carry bit from bit $0$ up to $M$.

### 4. `encode_hold(bdd, n_classes)`
Asserts `win_bar_l <-> win_l` for each class, while omitting any mention of `lc_bar_l_j` (leaving counters unconstrained).

### 5. `encode_subsequent_tree(bdd, forest, disc, t_idx, rho=None)`
Assembles $\Phi_t$:
1. Computes $\text{won} = \bigvee_\ell \text{win}_\ell$.
2. Loops over leaves of $T_t$, computing $\pi_b$ and choosing between $\text{Win}_s$ (if $\text{Clinch}_s^{(t)}$) and $\text{Inc}_s$ (if $\neg \text{Clinch}_s^{(t)}$).
3. Returns `(won & Hold) | (~won & Step_t)`.

