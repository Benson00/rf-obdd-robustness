from __future__ import annotations

import math
import random
from typing import Literal
import scipy.stats as stats
import dd.autoref as _bdd
from rfbdd.discretize import Discretization


SelectorName = Literal[
    "info_gain",
    "shapley",
    "p_value",
    "random_node",
    "modal_variable",
]


def get_free_input_vars(
    rho: dict[str, bool],
    disc: Discretization,
) -> list[str]:
    """Return all input bit variable names not yet fixed in rho."""
    return [name for name in disc.all_bit_names if name not in rho]


def count_nodes_per_input_variable(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    allowed_vars: list[str],
) -> dict[str, int]:
    """Count occurrences of each allowed input variable across all nodes in func's BDD."""
    counts = {var: 0 for var in allowed_vars}
    visited = set()
    stack = [abs(func.node)]

    while stack:
        u = stack.pop()
        if u in visited or u <= 1:
            continue
        visited.add(u)

        level, low, high = bdd._bdd._succ[u]
        var_of_u = bdd._bdd.var_at_level(level)
        if var_of_u in counts:
            counts[var_of_u] += 1

        stack.append(abs(low))
        stack.append(abs(high))

    return counts


def select_by_modal_variable(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    rho: dict[str, bool],
    disc: Discretization,
) -> str:
    """Return input variable labelling greatest number of nodes."""
    free_vars = get_free_input_vars(rho, disc)
    if not free_vars:
        raise RuntimeError("No free input bit available to split.")

    counts = count_nodes_per_input_variable(bdd, acc, free_vars)

    best_var = free_vars[0]
    max_count = -1

    for var in free_vars:
        c = counts[var]
        if c > max_count:
            max_count = c
            best_var = var

    return best_var


def select_by_random_node(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    rho: dict[str, bool],
    disc: Discretization,
) -> str:
    """sample an input variable proportionally to node occurrences."""
    free_vars = get_free_input_vars(rho, disc)
    if not free_vars:
        raise RuntimeError("No free input bit available to split.")

    counts = count_nodes_per_input_variable(bdd, acc, free_vars)
    total_occurrences = sum(counts.values())

    if total_occurrences == 0:
        return random.choice(free_vars)

    vars_list = list(counts.keys())
    weights = [counts[v] for v in vars_list]
    return random.choices(vars_list, weights=weights, k=1)[0]


def project_input_solution(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    disc: Discretization,
) -> _bdd.Function:
    """Compute input projection exists c.A by quantifying away state variables."""
    state_vars = [v for v in acc.support if v not in disc.all_bit_names]
    if not state_vars:
        return acc
    return bdd.exist(state_vars, acc)


def count_models_on_inputs(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    free_vars: list[str],
) -> int:
    """Compute the number of satisfying assignments over free_vars for an input projection."""
    if func == bdd.false:
        return 0
    if func == bdd.true:
        return 1 << len(free_vars)

    support = func.support.intersection(free_vars)
    base_count = int(bdd.count(func, nvars=len(support)))
    unconstrained = len(free_vars) - len(support)
    return base_count * (1 << max(0, unconstrained))


def _prepare_selection_target(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    free_vars: list[str],
    disc: Discretization,
) -> tuple[_bdd.Function, list[str], list[str]]:
    """Return (target_bdd, count_vars, eval_vars) for semantic selection on A."""
    state_vars = [v for v in acc.support if v not in disc.all_bit_names]

    # For small state support (<=15 vars), compute exact existential input projection.
    # For large state support (>15 vars), bdd.exist exhibits combinatorial explosion in CUDD;
    # per Remark 9, we evaluate directly on accumulator A without projecting.
    if len(state_vars) <= 15:
        f_proj = project_input_solution(bdd, acc, disc)
        if f_proj != bdd.true and f_proj != bdd.false:
            active = [v for v in free_vars if v in f_proj.support]
            return f_proj, free_vars, (active if active else free_vars)

    state_vars_sorted = sorted(
        state_vars,
        key=lambda v: bdd._bdd.level_of_var(v),
    )
    all_vars = free_vars + state_vars_sorted
    active = [v for v in free_vars if v in acc.support]
    return acc, all_vars, (active if active else free_vars)


def select_by_information_gain(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    rho: dict[str, bool],
    disc: Discretization,
) -> str:
    """Maximize binary entropy Hb(q_k) of the split on the solution set of A."""
    free_vars = get_free_input_vars(rho, disc)
    if not free_vars:
        raise RuntimeError("No free input bit available to split.")

    target, count_vars, eval_vars = _prepare_selection_target(bdd, acc, free_vars, disc)
    total_models = count_models_on_inputs(bdd, target, count_vars)
    if total_models <= 1:
        return eval_vars[0]

    best_var = eval_vars[0]
    best_entropy = -1.0

    for var_name in eval_vars:
        f_x1 = bdd.let({var_name: bdd.true}, target)
        sub_vars = [v for v in count_vars if v != var_name]
        n1 = count_models_on_inputs(bdd, f_x1, sub_vars)
        qk = n1 / total_models

        if qk <= 0.0 or qk >= 1.0:
            entropy = 0.0
        else:
            entropy = -qk * math.log2(qk) - (1.0 - qk) * math.log2(1.0 - qk)

        if entropy > best_entropy:
            best_entropy = entropy
            best_var = var_name

    return best_var


def _zero_stratified_input_poly(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    input_vars: list[str],
    state_vars: list[str],
) -> list[int]:
    """Compute zero-stratified polynomial P_g(t) = sum_z M_z(g) t^z over input_vars."""
    if func == bdd.false:
        return [0]
    n_in = len(input_vars)
    n_st = len(state_vars)
    if func == bdd.true:
        scale = 1 << n_st
        return [math.comb(n_in, z) * scale for z in range(n_in + 1)]

    in_idx = {v: i for i, v in enumerate(input_vars)}
    st_idx = {v: i for i, v in enumerate(state_vars)}

    st_memo: dict[int, int] = {}

    def count_state(u_node: int) -> int:
        u = abs(u_node)
        if u <= 1:
            return 1 if u_node == 1 else 0
        if u_node in st_memo:
            return st_memo[u_node]
        level, lo_raw, hi_raw = bdd._bdd._succ[u]
        var_name = bdd._bdd.var_at_level(level)
        cur_s = st_idx[var_name]
        lo = -lo_raw if u_node < 0 else lo_raw
        hi = -hi_raw if u_node < 0 else hi_raw

        def get_st_lev(child: int) -> int:
            cu = abs(child)
            if cu <= 1:
                return n_st
            return st_idx[bdd._bdd.var_at_level(bdd._bdd._succ[cu][0])]

        lo_s = get_st_lev(lo)
        hi_s = get_st_lev(hi)
        c_lo = count_state(lo) * (1 << (lo_s - cur_s - 1))
        c_hi = count_state(hi) * (1 << (hi_s - cur_s - 1))
        res = c_lo + c_hi
        st_memo[u_node] = res
        return res

    def state_entry_weight(u_node: int) -> int:
        u = abs(u_node)
        if u <= 1:
            return (1 << n_st) if u_node == 1 else 0
        s_lev = st_idx[bdd._bdd.var_at_level(bdd._bdd._succ[u][0])]
        return count_state(u_node) * (1 << s_lev)

    binom_cache: dict[int, list[int]] = {}

    def get_binom(p: int) -> list[int]:
        if p not in binom_cache:
            binom_cache[p] = [math.comb(p, k) for k in range(p + 1)]
        return binom_cache[p]

    def poly_mul_binom(poly: list[int], power: int, shift_t: bool) -> list[int]:
        if power == 0 and not shift_t:
            return list(poly)
        b = get_binom(power)
        offset = 1 if shift_t else 0
        out = [0] * (len(poly) + power + offset)
        for i, c1 in enumerate(poly):
            if c1:
                for j, c2 in enumerate(b):
                    out[i + j + offset] += c1 * c2
        return out

    poly_memo: dict[int, list[int]] = {}

    def visit_input(u_node: int) -> list[int]:
        u = abs(u_node)
        if u <= 1:
            return [state_entry_weight(u_node)]
        level, lo_raw, hi_raw = bdd._bdd._succ[u]
        var_name = bdd._bdd.var_at_level(level)
        if var_name not in in_idx:
            return [state_entry_weight(u_node)]
        if u_node in poly_memo:
            return poly_memo[u_node]

        cur_i = in_idx[var_name]
        lo = -lo_raw if u_node < 0 else lo_raw
        hi = -hi_raw if u_node < 0 else hi_raw

        def get_in_lev(child: int) -> int:
            cu = abs(child)
            if cu <= 1:
                return n_in
            cv = bdd._bdd.var_at_level(bdd._bdd._succ[cu][0])
            return in_idx.get(cv, n_in)

        lo_i = get_in_lev(lo)
        hi_i = get_in_lev(hi)
        p_lo = visit_input(lo)
        p_hi = visit_input(hi)

        term_lo = poly_mul_binom(p_lo, lo_i - cur_i - 1, shift_t=True)
        term_hi = poly_mul_binom(p_hi, hi_i - cur_i - 1, shift_t=False)

        max_len = max(len(term_lo), len(term_hi))
        res = [0] * max_len
        for idx, val in enumerate(term_lo):
            res[idx] += val
        for idx, val in enumerate(term_hi):
            res[idx] += val
        poly_memo[u_node] = res
        return res

    root_u = func.node
    cu = abs(root_u)
    if cu <= 1:
        r_lev = n_in
    else:
        rv = bdd._bdd.var_at_level(bdd._bdd._succ[cu][0])
        r_lev = in_idx.get(rv, n_in)

    return poly_mul_binom(visit_input(root_u), r_lev, shift_t=False)


def select_by_shapley_value(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    rho: dict[str, bool],
    disc: Discretization,
) -> str:
    """Return input variable with extremal Shapley value phi_k in coalition game."""
    free_vars = get_free_input_vars(rho, disc)
    if not free_vars:
        raise RuntimeError("No free input bit available to split.")

    target, count_vars, eval_vars = _prepare_selection_target(bdd, acc, free_vars, disc)
    total_models = count_models_on_inputs(bdd, target, count_vars)
    if total_models <= 1:
        return eval_vars[0]

    state_vars = [v for v in count_vars if v not in disc.all_bit_names]
    active_inputs = [v for v in free_vars if v in target.support]
    if not active_inputs:
        active_inputs = free_vars

    best_var = eval_vars[0]
    max_abs_phi = -1.0

    for var_name in eval_vars:
        gk = bdd.let({var_name: bdd.false}, target)
        sub_inputs = [v for v in active_inputs if v != var_name]
        pgk_sub = _zero_stratified_input_poly(bdd, gk, sub_inputs, state_vars)
        # In g_k = f & ~x_k, x_k is forced to 0 (+1 zero, shifting degree z -> z + 1)
        phi_k = 0.0
        for z_sub, coeff in enumerate(pgk_sub):
            if coeff:
                phi_k -= float(coeff) / (z_sub + 1)

        abs_phi = abs(phi_k)
        if abs_phi > max_abs_phi:
            max_abs_phi = abs_phi
            best_var = var_name

    return best_var


def safe_association_p_value(a_obs: int, total_assignments: int, k_solutions: int, m: int) -> float:
    """Compute one-sided Fisher exact / hypergeometric upper tail Pr[X >= a_obs]
    for X ~ Hyp(N, K, m), using exact hypergeometric summation for small K/N and
    finite-population normal tail when K and N are large.
    """
    if a_obs <= 0:
        return 1.0
    if a_obs > min(k_solutions, m):
        return 0.0
    if k_solutions >= total_assignments:
        return 1.0

    # Only call scipy's discrete hypergeometric C loop when universe N is small (<= 65,536)
    # where exact discrete summation is instantaneous (<0.1ms).
    # For larger N,  prescribes the normal approximation with
    # finite-population correction, avoiding scipy's hanging C loops on huge integers.
    if total_assignments <= 65_536:
        try:
            return float(stats.hypergeom.sf(a_obs - 1, total_assignments, k_solutions, m))
        except (OverflowError, MemoryError):
            pass

    if k_solutions <= 500:
        try:
            return float(stats.binom.sf(a_obs - 1, k_solutions, 0.5))
        except (OverflowError, MemoryError):
            pass

    # Exact hypergeometric variance with finite-population correction (Proposition 9):
    # Var(X) = K * (m/N) * (1 - m/N) * (N - K) / (N - 1) = 0.25 * K * (N - K) / (N - 1)
    fpc = float(total_assignments - k_solutions) / float(total_assignments - 1)
    sigma = 0.5 * math.sqrt(float(k_solutions) * max(1e-15, fpc))
    z = (float(a_obs) - 0.5 - 0.5 * float(k_solutions)) / sigma
    return float(stats.norm.sf(z))


def select_by_association_p_value(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    rho: dict[str, bool],
    disc: Discretization,
) -> str:
    """Minimize association p-value s_k = min(p_{k,1}, p_{k,0})."""
    free_vars = get_free_input_vars(rho, disc)
    if not free_vars:
        raise RuntimeError("No free input bit available to split.")

    target, count_vars, eval_vars = _prepare_selection_target(bdd, acc, free_vars, disc)
    n = len(count_vars)
    k_solutions = count_models_on_inputs(bdd, target, count_vars)
    if k_solutions <= 1 or n == 1:
        return eval_vars[0]

    total_assignments = 1 << n
    m = 1 << (n - 1)

    best_var = eval_vars[0]
    min_s = float("inf")

    for var_name in eval_vars:
        f_x1 = bdd.let({var_name: bdd.true}, target)
        sub_vars = [v for v in count_vars if v != var_name]
        a_k1 = count_models_on_inputs(bdd, f_x1, sub_vars)
        a_k0 = k_solutions - a_k1

        p_k1 = safe_association_p_value(a_k1, total_assignments, k_solutions, m)
        p_k0 = safe_association_p_value(a_k0, total_assignments, k_solutions, m)
        s_k = min(float(p_k1), float(p_k0))

        if s_k < min_s:
            min_s = s_k
            best_var = var_name

    return best_var


SELECTORS = {
    "info_gain": select_by_information_gain,
    "shapley": select_by_shapley_value,
    "p_value": select_by_association_p_value,
    "random_node": select_by_random_node,
    "modal_variable": select_by_modal_variable,
}


def get_selector(name: SelectorName | str):
    """Retrieve selector strategy callable by name."""
    if name not in SELECTORS:
        raise KeyError(f"Unknown selector '{name}'. Valid selectors: {list(SELECTORS.keys())}")
    return SELECTORS[name]
