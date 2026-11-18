from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import scipy.stats as stats
import dd.autoref as _bdd
from rfbdd.discretize import Discretization
from rfbdd.selectors import safe_association_p_value


def sat_count(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    nvars: int,
) -> int:
    """Return the exact number of satisfying assignments over nvars variables."""
    if func == bdd.false:
        return 0
    if func == bdd.true:
        return 1 << nvars

    supp_len = len(func.support)
    base = int(bdd.count(func, nvars=supp_len))
    unconstrained = nvars - supp_len
    return base * (1 << max(0, unconstrained))


def conditional_count(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    rho: dict[str, bool],
    nvars: int,
) -> tuple[int, int]:
    """Return (N_yes, N_no) under partial assignment rho."""
    if not rho:
        total = sat_count(bdd, func, nvars)
        return total, 0

    sub_map = {k: bdd.true if v else bdd.false for k, v in rho.items()}
    f_res = bdd.let(sub_map, func)
    free_vars = nvars - len(rho)
    n_yes = sat_count(bdd, f_res, free_vars)
    total = sat_count(bdd, func, nvars)
    n_no = total - n_yes
    return n_yes, n_no


def certified_volume(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    disc: Discretization,
    data_bounds: dict[int, tuple[float, float]] | None = None,
) -> float:
    """Compute the normalized real-space volume / measure of func."""
    if func == bdd.false:
        return 0.0

    n_features = disc.n_features
    # Compute normalized cell widths for each feature
    cell_weights: list[list[float]] = []
    for f in range(n_features):
        n_cells = disc.n_cells[f]
        thresholds = disc.thresholds[f]
        if n_cells <= 1:
            cell_weights.append([1.0])
            continue

        if data_bounds and f in data_bounds:
            f_min, f_max = data_bounds[f]
        else:
            f_min = float(thresholds[0] - 1.0)
            f_max = float(thresholds[-1] + 1.0)

        total_span = max(1e-9, f_max - f_min)
        lengths: list[float] = []
        # Cell 0: [f_min, thresh[0]]
        lengths.append(max(0.0, float(thresholds[0]) - f_min))
        # Interior cells: [thresh[i-1], thresh[i]]
        for i in range(1, len(thresholds)):
            lengths.append(max(0.0, float(thresholds[i]) - float(thresholds[i - 1])))
        # Last cell: [thresh[-1], f_max]
        lengths.append(max(0.0, f_max - float(thresholds[-1])))

        weights = [l / total_span for l in lengths]
        cell_weights.append(weights)

    # Sum product of cell weights over all satisfying cell configurations
    input_vars = disc.all_bit_names
    state_vars = [v for v in func.support if v not in input_vars]
    f_proj = bdd.exist(state_vars, func) if state_vars else func

    total_vol = 0.0
    for model in bdd.pick_iter(f_proj, care_vars=input_vars):
        # Decode cell index per feature
        prod = 1.0
        for f in range(n_features):
            b_f = disc.n_bits[f]
            if b_f == 0:
                continue
            idx = 0
            for j in range(b_f):
                bit_val = model[disc.bit_names[f][j]]
                if bit_val:
                    idx |= (1 << j)
            # Clamping 
            cell_idx = min(idx, disc.n_cells[f] - 1)
            surplus = (1 << b_f) - disc.n_cells[f]
            if cell_idx == disc.n_cells[f] - 1 and surplus > 0:
                prod *= (cell_weights[f][cell_idx] / (1 + surplus))
            else:
                prod *= cell_weights[f][cell_idx]
        total_vol += prod

    return total_vol


def compute_information_gain_ranking(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    disc: Discretization,
) -> list[tuple[str, float]]:
    """Rank input bits by binary entropy H_b(q_k) of the solution split."""
    input_vars = disc.all_bit_names
    state_vars = [v for v in func.support if v not in input_vars]
    f_proj = bdd.exist(state_vars, func) if state_vars else func

    total_k = sat_count(bdd, f_proj, len(input_vars))
    if total_k == 0:
        return [(v, 0.0) for v in input_vars]

    scores: list[tuple[str, float]] = []
    for var in input_vars:
        f_x1 = bdd.let({var: bdd.true}, f_proj)
        n1 = sat_count(bdd, f_x1, len(input_vars) - 1)
        qk = n1 / total_k
        if qk <= 0.0 or qk >= 1.0:
            h = 0.0
        else:
            h = -qk * math.log2(qk) - (1.0 - qk) * math.log2(1.0 - qk)
        scores.append((var, h))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def _binom_coeffs(power: int) -> np.ndarray:
    """Return binomial coefficients of (1 + t)^power."""
    return np.array([math.comb(power, k) for k in range(power + 1)], dtype=object)


def _compute_zero_stratified_polynomial(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    active_vars: Sequence[str],
) -> np.ndarray:
    """Compute zero-stratified polynomial P_g(t) = sum_z M_z(g) t^z."""
    n_vars = len(active_vars)
    v2l = {v: i for i, v in enumerate(active_vars)}
    memo: dict[int, np.ndarray] = {}

    def visit(u_node: int) -> np.ndarray:
        u = abs(u_node)
        if u <= 1:
            return np.array([1], dtype=object) if u_node == 1 else np.array([0], dtype=object)
        if u_node in memo:
            return memo[u_node]

        level, lo_raw, hi_raw = bdd._bdd._succ[u]
        var_name = bdd._bdd.var_at_level(level)
        lo = -lo_raw if u_node < 0 else lo_raw
        hi = -hi_raw if u_node < 0 else hi_raw

        cur_lev = v2l[var_name]

        def get_child_level(child: int) -> int:
            cu = abs(child)
            if cu <= 1:
                return n_vars
            return v2l[bdd._bdd.var_at_level(bdd._bdd._succ[cu][0])]

        lo_lev = get_child_level(lo)
        hi_lev = get_child_level(hi)

        p_lo = visit(lo)
        p_hi = visit(hi)

        # Low edge: t * (1+t)^(lo_lev - cur_lev - 1) * p_lo
        lo_factor = np.convolve(_binom_coeffs(lo_lev - cur_lev - 1), np.array([0, 1], dtype=object))
        term_lo = np.convolve(p_lo, lo_factor)

        # High edge: (1+t)^(hi_lev - cur_lev - 1) * p_hi
        hi_factor = _binom_coeffs(hi_lev - cur_lev - 1)
        term_hi = np.convolve(p_hi, hi_factor)

        max_len = max(len(term_lo), len(term_hi))
        res = np.pad(term_lo, (0, max_len - len(term_lo))) + np.pad(term_hi, (0, max_len - len(term_hi)))
        memo[u_node] = res
        return res

    root_u = func.node
    if abs(root_u) <= 1:
        return _binom_coeffs(n_vars) if root_u == 1 else np.array([0], dtype=object)

    r_lev = v2l[bdd._bdd.var_at_level(bdd._bdd._succ[abs(root_u)][0])]
    prefactor = _binom_coeffs(r_lev)
    p_g = np.convolve(prefactor, visit(root_u))
    return p_g


def compute_shapley_ranking(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    disc: Discretization,
) -> list[tuple[str, float]]:
    """Compute exact Shapley values phi_k for all input bits."""
    input_vars = disc.all_bit_names
    state_vars = [v for v in func.support if v not in input_vars]
    f_proj = bdd.exist(state_vars, func) if state_vars else func

    if f_proj == bdd.false:
        return [(v, 0.0) for v in input_vars]

    scores: list[tuple[str, float]] = []
    for var in input_vars:
        # Restriction forcing x_k = 0: g_k = f & ~var
        gk = bdd.let({var: bdd.false}, f_proj)
        pgk = _compute_zero_stratified_polynomial(bdd, gk, input_vars)
        # phi_k = - sum_{z >= 1} M_z(g_k) / z
        phi_k = 0.0
        for z in range(1, len(pgk)):
            phi_k -= float(pgk[z]) / z
        scores.append((var, phi_k))

    # Rank by magnitude |phi_k| descending (Remark 10)
    scores.sort(key=lambda item: abs(item[1]), reverse=True)
    return scores


def compute_association_p_value_ranking(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    disc: Discretization,
    bonferroni: bool = True,
) -> list[tuple[str, float, int]]:
    """Rank input bits by association-rule p-value s_k = min(p_k1, p_k0)."""
    input_vars = disc.all_bit_names
    state_vars = [v for v in func.support if v not in input_vars]
    f_proj = bdd.exist(state_vars, func) if state_vars else func

    n = len(input_vars)
    k_solutions = sat_count(bdd, f_proj, n)
    if k_solutions <= 1 or n == 1:
        return [(v, 1.0, 1) for v in input_vars]

    total_assignments = 1 << n
    m = 1 << (n - 1)

    scores: list[tuple[str, float, int]] = []
    for var in input_vars:
        f_x1 = bdd.let({var: bdd.true}, f_proj)
        a_k1 = sat_count(bdd, f_x1, n - 1)
        a_k0 = k_solutions - a_k1

        p_k1 = safe_association_p_value(a_k1, total_assignments, k_solutions, m)
        p_k0 = safe_association_p_value(a_k0, total_assignments, k_solutions, m)

        if p_k1 <= p_k0:
            best_s = float(p_k1)
            best_b = 1
        else:
            best_s = float(p_k0)
            best_b = 0

        if bonferroni:
            best_s = min(1.0, 2 * n * best_s)

        scores.append((var, best_s, best_b))

    # Rank by p-value ascending (most significant first)
    scores.sort(key=lambda item: item[1])
    return scores
