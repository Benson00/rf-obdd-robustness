from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import dd.autoref as _bdd
from rfbdd.forest import Forest
from rfbdd.discretize import Discretization
from rfbdd.first_tree import (
    compute_counter_width_m,
    counter_bit_name,
    win_flag_name,
    encode_first_tree,
)
from rfbdd.subsequent_tree import encode_subsequent_tree
from rfbdd.composition import setup_composition_manager


@dataclass
class FrontierElement:
    """An active branch in the frontier."""

    rho: dict[str, bool]       # partial assignment fixing input bits
    acc: _bdd.Function         # current accumulator cofactored by rho
    t: int                     # index of next tree to compose


@dataclass
class LeafNode:
    """A completed frontier branch reaching the last tree."""

    rho: dict[str, bool]       # cube assignment partitioning input space
    acc: _bdd.Function         # final accumulator Acc_{D-1}|_rho


@dataclass
class DecompositionStats:
    """Metrics recorded during BudgetedForest decomposition."""

    peak_acc_size: int         # max |A| encountered across all steps
    peak_manager_size: int     # max |G| (total live nodes in BDD manager)
    num_leaves: int            # number of final leaf nodes
    runtime: float             # elapsed wall-clock seconds
    num_splits: int            # total number of case-splits performed
    completed: bool            # True if all branches reached last tree D


def encode_cube(bdd: _bdd.BDD, rho: dict[str, bool]) -> _bdd.Function:
    """Convert a dictionary assignment rho into a BDD cube conjunction."""
    cube = bdd.true
    for var_name, val in rho.items():
        var = bdd.var(var_name)
        cube &= var if val else ~var
    return cube


def default_first_free_selector(
    bdd: _bdd.BDD,
    acc: _bdd.Function,
    rho: dict[str, bool],
    disc: Discretization,
) -> str | None:
    """Baseline fallback selector picking the first free input bit, or None if all are fixed."""
    for name in disc.all_bit_names:
        if name not in rho:
            return name
    return None


class _BudgetTimeout(Exception):
    """Internal signal raised when elapsed time exceeds timeout_seconds."""


def budgeted_forest(
    forest: Forest,
    disc: Discretization,
    budget: int,
    selector: Callable[[_bdd.BDD, _bdd.Function, dict[str, bool], Discretization], str | None] | None = None,
    return_stats: bool = False,
    timeout_seconds: float | None = None,
    max_leaves: int | None = None,
) -> tuple[_bdd.BDD, list[LeafNode]] | tuple[_bdd.BDD, list[LeafNode], DecompositionStats]:
    """Execute BudgetedForest decomposition under budget B."""
    import time
    t_start = time.perf_counter()
    deadline = (t_start + timeout_seconds) if timeout_seconds is not None else None

    if selector is None:
        selector = default_first_free_selector

    bdd, current_vars, barred_vars, rename_map = setup_composition_manager(forest, disc)
    sub_map = {k: bdd.var(v) for k, v in rename_map.items()}

    orig_find_or_add = bdd._bdd.find_or_add
    if deadline is not None:
        op_count = 0

        def _guarded_find_or_add(i: int, v: int, w: int) -> int:
            nonlocal op_count
            op_count += 1
            if (op_count & 255) == 0 and time.perf_counter() > deadline:
                raise _BudgetTimeout()
            return orig_find_or_add(i, v, w)

        bdd._bdd.find_or_add = _guarded_find_or_add

    peak_acc_size = 0
    peak_manager_size = len(bdd)
    num_splits = 0
    completed = True
    final_leaves: list[LeafNode] = []
    frontier: list[FrontierElement] = []

    try:
        # Base case: Tree T_0 produces initial accumulator Acc_0
        phi_0 = encode_first_tree(bdd, forest, disc, rho=None, barred=True)
        acc_0 = bdd.let(sub_map, phi_0)

        peak_acc_size = len(acc_0)
        peak_manager_size = len(bdd)

        # Frontier initially contains the full unconstrained space
        frontier.append(FrontierElement(rho={}, acc=acc_0, t=1))

        while frontier:
            # Check resource limits (timeout or max leaves limit)
            if deadline is not None and time.perf_counter() > deadline:
                completed = False
                break
            if max_leaves is not None and (len(final_leaves) + len(frontier)) > max_leaves:
                completed = False
                break

            elem = frontier.pop()
            rho, acc, t = elem.rho, elem.acc, elem.t

            # Grow within budget
            has_free_bits = any(name not in rho for name in disc.all_bit_names)
            while t < forest.D and (len(acc) <= budget or not has_free_bits):
                if deadline is not None and time.perf_counter() > deadline:
                    completed = False
                    break
                phi_t = encode_subsequent_tree(bdd, forest, disc, t_idx=t, rho=rho)
                prod = acc & phi_t
                quantified = bdd.exist(current_vars, prod)
                acc = bdd.let(sub_map, quantified)
                peak_acc_size = max(peak_acc_size, len(acc))
                peak_manager_size = max(peak_manager_size, len(bdd))
                t += 1

            if not completed:
                break

            if t == forest.D:
                # All trees collapsed -> save leaf
                final_leaves.append(LeafNode(rho=rho, acc=acc))
            else:
                # Budget exceeded (|acc| > budget) with trees remaining and free bits available
                x = selector(bdd, acc, rho, disc)
                if deadline is not None and time.perf_counter() > deadline:
                    completed = False
                    break
                if x is None:
                    # All input bits already fixed -> cannot split on inputs, let it complete
                    while t < forest.D:
                        if deadline is not None and time.perf_counter() > deadline:
                            completed = False
                            break
                        phi_t = encode_subsequent_tree(bdd, forest, disc, t_idx=t, rho=rho)
                        prod = acc & phi_t
                        quantified = bdd.exist(current_vars, prod)
                        acc = bdd.let(sub_map, quantified)
                        peak_acc_size = max(peak_acc_size, len(acc))
                        peak_manager_size = max(peak_manager_size, len(bdd))
                        t += 1
                    if completed:
                        final_leaves.append(LeafNode(rho=rho, acc=acc))
                    else:
                        break
                    continue

                num_splits += 1

                # Left child: rho & x=1
                acc_1 = bdd.let({x: bdd.true}, acc)
                rho_1 = dict(rho)
                rho_1[x] = True
                frontier.append(FrontierElement(rho=rho_1, acc=acc_1, t=t))

                # Right child: rho & x=0
                acc_0 = bdd.let({x: bdd.false}, acc)
                rho_0 = dict(rho)
                rho_0[x] = False
                frontier.append(FrontierElement(rho=rho_0, acc=acc_0, t=t))

                peak_acc_size = max(peak_acc_size, len(acc_1), len(acc_0))
                peak_manager_size = max(peak_manager_size, len(bdd))
    except _BudgetTimeout:
        completed = False
    finally:
        if deadline is not None:
            bdd._bdd.find_or_add = orig_find_or_add

    runtime = time.perf_counter() - t_start

    if return_stats:
        stats = DecompositionStats(
            peak_acc_size=peak_acc_size,
            peak_manager_size=peak_manager_size,
            num_leaves=len(final_leaves) + len(frontier),
            runtime=runtime,
            num_splits=num_splits,
            completed=completed,
        )
        return bdd, final_leaves, stats

    return bdd, final_leaves


def extract_frontier_winning_regions(
    bdd: _bdd.BDD,
    leaves: list[LeafNode],
    forest: Forest,
) -> dict[int, _bdd.Function]:
    """Reconstruct Win_l(x) = OR_rho (chi_rho & exist_c (A_rho & win_l))."""
    m = compute_counter_width_m(forest.D)
    all_state_vars: list[str] = []
    for l in range(forest.n_classes):
        for j in range(m + 1):
            all_state_vars.append(counter_bit_name(l, j, barred=False))
        all_state_vars.append(win_flag_name(l, barred=False))

    win_bdds: dict[int, _bdd.Function] = {l: bdd.false for l in range(forest.n_classes)}

    for leaf in leaves:
        chi_rho = encode_cube(bdd, leaf.rho)
        for l in range(forest.n_classes):
            acc_l = leaf.acc & bdd.var(win_flag_name(l, barred=False))
            w_rho = bdd.exist(all_state_vars, acc_l)
            win_bdds[l] |= (chi_rho & w_rho)

    return win_bdds