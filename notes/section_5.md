# Section 5: Encoding a Node Test

---

## 1. Goal

Each internal node of a tree tests:
$$x_f \le \text{threshold}$$
Under Section 3's lossless discretization, this is identical to:
$$X_f \le \tau$$
where $\tau \in \{0, \dots, \text{cells}(f) - 1\}$ is the threshold's cell index, and:
$$X_f = \sum_{j=0}^{b_f - 1} 2^j B_f^j$$
is the integer encoded by the $b_f$ bits of feature $f$.

**The Challenge:**
Avoid a quadratic "pyramid" formula of the form:
$$(X_f = 0) \vee (X_f = 1) \vee \dots \vee (X_f = \tau)$$
which would explode in size for large cell counts.

---

## 2. Theory

### Definition 4 (Less-or-Equal Comparator)
The test $X_f \le \tau$ is written as a **linear formula** $\psi_{b_f - 1}$ by comparing $X_f$ against the fixed constant $\tau = \sum_{j=0}^{b_f - 1} 2^j \tau_j$ bit by bit:

$$\psi_{-1} = \top$$
$$\psi_j = \begin{cases} 
\neg B_f^j \wedge \psi_{j-1} & \text{if } \tau_j = 0 \\
\neg B_f^j \vee \psi_{j-1} & \text{if } \tau_j = 1
\end{cases} \quad (0 \le j \le b_f - 1)$$

The node test formula is denoted:
$$\text{nt}(f, \tau) := \psi_{b_f - 1}$$

### Why the Recurrence Works
Processing from the most significant bit downwards:
1. **If $\tau_j = 0$**: The sample bit $B_f^j$ **must be 0** ($\neg B_f^j$), otherwise $X_f$ would already exceed $\tau$ at position $j$. The lower bits must still satisfy $\psi_{j-1}$.
2. **If $\tau_j = 1$**:
   - If $B_f^j = 0$, $X_f$ is strictly smaller than $\tau$ at position $j$ (lower bits do not matter).
   - If $B_f^j = 1$, it matches $\tau_j$, so lower bits must satisfy $\psi_{j-1}$.
   - Simplifying: $\neg B_f^j \vee (B_f^j \wedge \psi_{j-1}) \equiv \neg B_f^j \vee \psi_{j-1}$.

---

## 3. Step-by-Step Code Walkthrough

Let's test whether $X_f \le 1$ ($\tau = 1$) on a feature with 2 bits ($b_f = 2$):
- Bit 0 (`B_0`): weight $2^0 = 1$
- Bit 1 (`B_1`): weight $2^1 = 2$
- Threshold $\tau = 1$: in binary, its bit 0 is $1$ ($\tau_0 = 1$) and its bit 1 is $0$ ($\tau_1 = 0$).

---

### Case A: Evaluating a Sample with `eval_node_test_formula`

#### Sample 1: Input $X_f = 1$ (`bits = [True, False]`)
Value: $(1 \cdot 2^0) + (0 \cdot 2^1) = 1$. Expected: $1 \le 1 \implies \text{True}$.

Execution trace:
1. **Init**: `psi = True` ($\psi_{-1} = \top$).
2. **Iteration $j = 0$ (Bit 0)**:
   - Threshold bit: $\tau_0 = 1$.
   - Sample bit: `bits[0] = True` $\implies$ negated is `not_b = False`.
   - Since $\tau_0 == 1$, apply OR:
     $$\text{psi} = \text{not\_b} \vee \text{psi} = \text{False} \vee \text{True} = \mathbf{True}$$
3. **Iteration $j = 1$ (Bit 1)**:
   - Threshold bit: $\tau_1 = 0$.
   - Sample bit: `bits[1] = False` $\implies$ negated is `not_b = True`.
   - Since $\tau_1 == 0$, apply AND:
     $$\text{psi} = \text{not\_b} \wedge \text{psi} = \text{True} \wedge \text{True} = \mathbf{True}$$
4. **Result**: returns `True`.

---

#### Sample 2: Input $X_f = 2$ (`bits = [False, True]`)
Value: $(0 \cdot 2^0) + (1 \cdot 2^1) = 2$. Expected: $2 \le 1 \implies \text{False}$.

Execution trace:
1. **Init**: `psi = True`.
2. **Iteration $j = 0$ (Bit 0)**:
   - $\tau_0 = 1$, `not_b = True` $\implies \text{psi} = \text{True} \vee \text{True} = \mathbf{True}$.
3. **Iteration $j = 1$ (Bit 1)**:
   - Threshold bit: $\tau_1 = 0$.
   - Sample bit: `bits[1] = True` $\implies$ negated is `not_b = False`.
   - Since $\tau_1 == 0$, apply AND:
     $$\text{psi} = \text{not\_b} \wedge \text{psi} = \text{False} \wedge \text{True} = \mathbf{False}$$
4. **Result**: returns `False`.

---

### Case B: Building the OBDD Node with `encode_node_test`

`encode_node_test` executes the exact same recurrence, but using **symbolic BDD operations**:
- **Init**: `psi = bdd.true`.
- **Iteration $j = 0$ ($\tau_0 = 1$)**:
  $$\text{psi} = \neg B_f^0 \vee \text{bdd.true} = \mathbf{bdd.true}$$
- **Iteration $j = 1$ ($\tau_1 = 0$)**:
  $$\text{psi} = \neg B_f^1 \wedge \text{bdd.true} = \mathbf{\neg B_f^1}$$

The resulting OBDD formula is simply $\neg B_f^1$:
- Any input with $B_f^1 = 0$ (cells 0 and 1) evaluates to `True` ($X_f \le 1$).
- Any input with $B_f^1 = 1$ (cells 2 and 3) evaluates to `False` ($X_f > 1$).

### Case B: Building the OBDD Node with `encode_node_test`

`encode_node_test` executes the exact same recurrence, but using **symbolic BDD operations**:
- **Init**: `psi = bdd.true`.
- **Iteration $j = 0$ ($\tau_0 = 1$)**:
  $$\text{psi} = \neg B_f^0 \vee \text{bdd.true} = \mathbf{bdd.true}$$
- **Iteration $j = 1$ ($\tau_1 = 0$)**:
  $$\text{psi} = \neg B_f^1 \wedge \text{bdd.true} = \mathbf{\neg B_f^1}$$

The resulting OBDD formula is simply $\neg B_f^1$:
- Any input with $B_f^1 = 0$ (cells 0 and 1) evaluates to `True` ($X_f \le 1$).
- Any input with $B_f^1 = 1$ (cells 2 and 3) evaluates to `False` ($X_f > 1$).

