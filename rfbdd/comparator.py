from __future__ import annotations

import dd.autoref as _bdd
from rfbdd.discretize import Discretization


def eval_node_test_formula(bits: list[bool], tau: int) -> bool:
    """Evaluate the comparator formula on concrete boolean bit values.

    Example:
    --------
    Testing X_f <= 1 (tau = 1) with 2 bits [B_0, B_1]:
    - bits = [True, False] represents integer 1 (01_2 where _2 means base 2): 
      1 <= 1 -> returns True
    - bits = [False, True] represents integer 2 (10_2):
      2 <= 1 -> returns False
    """
    bf = len(bits)
    # Base case: psi_{-1} = True
    psi = True

    # Scan bits from position 0 up to bf - 1
    for j in range(bf):
        # Extract j-th bit of threshold tau: tau_j in {0, 1}
        tau_j = (tau >> j) & 1
        # Negate sample bit: ~B_f^j
        not_b = not bits[j]

        # Recurrence: (~B_f^j AND psi_{j-1}) if tau_j = 0, else (~B_f^j OR psi_{j-1})
        psi = (not_b and psi) if tau_j == 0 else (not_b or psi)

    return psi


def encode_node_test(
    bdd: _bdd.BDD,
    disc: Discretization,
    f: int,
    tau: int,
    rho: dict[str, bool] | None = None,
):
    """Build the BDD for the node test nt(f, tau) := psi_{b_f - 1}.

    Example:
    --------
    Feature 0 has 4 cells (2 bits: 'x0_0', 'x0_1') and threshold index tau = 1:
        nt = encode_node_test(bdd, disc, f=0, tau=1)
    
    Evaluating the resulting BDD:
    - assignment {'x0_0': True, 'x0_1': False} (cell 1) -> bdd.let(assign, nt) == bdd.true
    - assignment {'x0_0': False, 'x0_1': True} (cell 2) -> bdd.let(assign, nt) == bdd.false
    """
    bf = disc.n_bits[f]
    # If feature has 0 bits (only 1 cell), threshold test is trivially true
    if bf == 0:
        return bdd.true

    names = disc.bit_names[f]
    # Base case: psi_{-1} = True (terminal 1 in BDD)
    psi = bdd.true

    # Build formula iteratively from j = 0 to bf - 1
    for j in range(bf):
        var_name = names[j]
        # Extract j-th bit of tau
        tau_j = (tau >> j) & 1

        # Check if variable is fixed by partial assignment rho
        if rho is not None and var_name in rho:
            bit_val = rho[var_name]
            not_b = bdd.false if bit_val else bdd.true
        else:
            # Create negated BDD variable ~B_f^j
            not_b = ~bdd.var(var_name)

        # Apply recurrence with BDD operations
        if tau_j == 0:
            psi = not_b & psi  # Conjunction (~B_f^j AND psi_{j-1})
        else:
            psi = not_b | psi  # Disjunction (~B_f^j OR psi_{j-1})

    # Return final comparator BDD
    return psi
