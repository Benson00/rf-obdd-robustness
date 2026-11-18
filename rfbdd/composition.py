from __future__ import annotations

import dd.autoref as _bdd
import dd.bdd as _raw_bdd
from rfbdd.forest import Forest
from rfbdd.discretize import Discretization
from rfbdd.first_tree import (
    compute_counter_width_m,
    counter_bit_name,
    win_flag_name,
    encode_first_tree,
)
from rfbdd.subsequent_tree import encode_subsequent_tree

# Disable dd.bdd.BDD's slow debug __del__ (which runs collect_garbage, inspect.stack,
# and pprint.pformat over all tables when a manager is garbage-collected).
_raw_bdd.BDD.__del__ = lambda self: None


def _fast_quantify(
    self: _raw_bdd.BDD,
    u: int,
    j: int,
    ordvar: list[int],
    qvars: set[int],
    forall: bool,
    cache: dict[int, int],
) -> int:
    """Short-circuiting existential/universal quantification on OBDD nodes."""
    if abs(u) == 1:
        return u
    if u in cache:
        return cache[u]
    i, v, w = self._succ[abs(u)]
    if u < 0:
        v, w = -v, -w
    n = len(ordvar)
    while j < n:
        if ordvar[j] < i:
            j += 1
        else:
            break
    else:
        return u

    if i in qvars:
        if not forall and (v == 1 or w == 1):
            cache[u] = 1
            return 1
        if forall and (v == -1 or w == -1):
            cache[u] = -1
            return -1

    p = self._quantify(v, j, ordvar, qvars, forall, cache)
    if i in qvars:
        if not forall and p == 1:
            cache[u] = 1
            return 1
        if forall and p == -1:
            cache[u] = -1
            return -1

    q = self._quantify(w, j, ordvar, qvars, forall, cache)
    if i in qvars:
        r = self.ite(p, q, -1) if forall else self.ite(p, 1, q)
    else:
        r = self.find_or_add(i, p, q)
    cache[u] = r
    return r


_raw_bdd.BDD._quantify = _fast_quantify


def setup_composition_manager(
    forest: Forest,
    disc: Discretization,
) -> tuple[_bdd.BDD, list[str], list[str], dict[str, str]]:
    """Declare variables in optimal order and build state variable lists and mappings.

    Order:
    1. Input bit variables x_f_j on top.
    2. Interleaved state variables: lc_l_j, lc_bar_l_j, ..., win_l, win_bar_l.
    """
    bdd = _bdd.BDD()
    m = compute_counter_width_m(forest.D)
    n_classes = forest.n_classes

    # Input variables on top
    for var_name in disc.all_bit_names:
        bdd.declare(var_name)

    # Interleaved state variables
    current_vars: list[str] = []
    barred_vars: list[str] = []
    rename_map: dict[str, str] = {}

    for l in range(n_classes):
        for j in range(m + 1):
            curr_c = counter_bit_name(l, j, barred=False)
            bar_c = counter_bit_name(l, j, barred=True)
            bdd.declare(curr_c)
            bdd.declare(bar_c)
            current_vars.append(curr_c)
            barred_vars.append(bar_c)
            rename_map[bar_c] = curr_c

        curr_w = win_flag_name(l, barred=False)
        bar_w = win_flag_name(l, barred=True)
        bdd.declare(curr_w)
        bdd.declare(bar_w)
        current_vars.append(curr_w)
        barred_vars.append(bar_w)
        rename_map[bar_w] = curr_w

    return bdd, current_vars, barred_vars, rename_map


def compose_forest(
    forest: Forest,
    disc: Discretization,
    rho: dict[str, bool] | None = None,
) -> tuple[_bdd.BDD, _bdd.Function]:
    """Incrementally compose all trees into the final accumulator Acc_{D-1}."""
    bdd, current_vars, barred_vars, rename_map = setup_composition_manager(forest, disc)

    # Base case: Tree T_0 produces barred state variables
    phi_0 = encode_first_tree(bdd, forest, disc, rho=rho, barred=True)

    # Acc_0 = Phi_0{bar -> current}
    sub_map = {k: bdd.var(v) for k, v in rename_map.items()}
    acc = bdd.let(sub_map, phi_0)

    # Inductive step: compose subsequent trees T_1 ... T_{D-1}
    for t in range(1, forest.D):
        phi_t = encode_subsequent_tree(bdd, forest, disc, t_idx=t, rho=rho)

        # Relational product: exist_c (Acc_{t-1} & Phi_t)
        prod = acc & phi_t
        quantified = bdd.exist(current_vars, prod)

        # Rename barred next-state variables back to current: {c_bar -> c}
        acc = bdd.let(sub_map, quantified)

    return bdd, acc


def extract_winning_regions(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    forest: Forest,
) -> dict[int, _bdd.Function]:
    """Extract Win_l(x) = exist_c (Acc_{D-1}(x, c) & label_win^l) for each class label."""
    m = compute_counter_width_m(forest.D)
    all_state_vars: list[str] = []

    for l in range(forest.n_classes):
        for j in range(m + 1):
            all_state_vars.append(counter_bit_name(l, j, barred=False))
        all_state_vars.append(win_flag_name(l, barred=False))

    win_bdds: dict[int, _bdd.Function] = {}
    for l in range(forest.n_classes):
        # Conjoin with label_win^l and quantify away all state variables c
        acc_with_winner = acc & bdd.var(win_flag_name(l, barred=False))
        win_bdds[l] = bdd.exist(all_state_vars, acc_with_winner)

    return win_bdds

