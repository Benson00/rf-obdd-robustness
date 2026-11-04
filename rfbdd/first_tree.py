from __future__ import annotations

import math
import dd.autoref as _bdd
from rfbdd.forest import Forest, Tree
from rfbdd.discretize import Discretization
from rfbdd.comparator import encode_node_test


def compute_counter_width_m(d: int) -> int:
    """Compute counter bit width parameter M = ceil(log2(D / 2)) + 1."""
    if d <= 1:
        return 0
    return math.ceil(math.log2(d / 2)) + 1


def counter_bit_name(label: int, j: int, barred: bool = False) -> str:
    """Return variable name for j-th bit of counter of class label."""
    prefix = "lc_bar" if barred else "lc"
    return f"{prefix}_{label}_{j}"


def win_flag_name(label: int, barred: bool = False) -> str:
    """Return variable name for win flag of class label."""
    prefix = "win_bar" if barred else "win"
    return f"{prefix}_{label}"


def declare_counter_and_win_vars(
    bdd: _bdd.BDD,
    n_classes: int,
    m: int,
    with_barred: bool = False,
) -> None:
    """Declare BDD variables for counters and win flags across all classes."""
    for l in range(n_classes):
        for j in range(m + 1):
            bdd.declare(counter_bit_name(l, j, barred=False))
            if with_barred:
                bdd.declare(counter_bit_name(l, j, barred=True))
        bdd.declare(win_flag_name(l, barred=False))
        if with_barred:
            bdd.declare(win_flag_name(l, barred=True))


def encode_counter_equals(
    bdd: _bdd.BDD,
    label: int,
    val: int,
    m: int,
    barred: bool = False,
):
    """Encode the conjunction (LC^l = val) fixing each counter bit."""
    expr = bdd.true
    for j in range(m + 1):
        v_j = (val >> j) & 1
        bit_var = bdd.var(counter_bit_name(label, j, barred=barred))
        expr &= bit_var if v_j == 1 else ~bit_var
    return expr


def encode_path_formula(
    bdd: _bdd.BDD,
    disc: Discretization,
    steps: list[tuple[int, int, int]],
    taus: dict[int, int],
    rho: dict[str, bool] | None = None,
):
    """Encode path formula pi_b as conjunction of node tests along the path."""
    pi_b = bdd.true
    for node, f, branch_dir in steps:
        tau = taus[node]
        nt = encode_node_test(bdd, disc, f=f, tau=tau, rho=rho)
        pi_b &= nt if branch_dir == 1 else ~nt
    return pi_b


def encode_first_tree(
    bdd: _bdd.BDD,
    forest: Forest,
    disc: Discretization,
    rho: dict[str, bool] | None = None,
    barred: bool = False,
):
    """Encode formula Phi_0 for tree T_0 initializing counters and win flags."""
    t0 = forest.trees[0]
    taus_0 = disc.taus[0]
    n_classes = forest.n_classes
    m = compute_counter_width_m(forest.D)

    # Disjunction over all leaves of T_0
    tree_body = bdd.false
    for steps, leaf_label in t0.paths():
        # Path condition pi_b
        pi_b = encode_path_formula(bdd, disc, steps, taus_0, rho=rho)

        # Counter of leaf label set to 1
        c_winner = encode_counter_equals(bdd, leaf_label, 1, m, barred=barred)

        # Counters of all other labels set to 0
        c_others = bdd.true
        for other_l in range(n_classes):
            if other_l != leaf_label:
                c_others &= encode_counter_equals(bdd, other_l, 0, m, barred=barred)

        tree_body |= (pi_b & c_winner & c_others)

    # All win flags forced to false (for D >= 2, single vote cannot win)
    no_win = bdd.true
    for l in range(n_classes):
        no_win &= ~bdd.var(win_flag_name(l, barred=barred))

    return tree_body & no_win

