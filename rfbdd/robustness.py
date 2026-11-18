from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import dd.autoref as _bdd
from rfbdd.discretize import Discretization
from rfbdd.first_tree import win_flag_name
from rfbdd.frontier import LeafNode


def sample_to_bits(
    sample: Sequence[float] | np.ndarray,
    disc: Discretization,
) -> dict[str, bool]:
    """Convert a continuous real sample vector into a binary cell-index bit assignment."""
    return disc.encode_sample(np.asarray(sample, dtype=float))


def bits_to_sample_cells(
    bits: dict[str, bool],
    disc: Discretization,
) -> list[int]:
    """Decode a binary bit assignment into discrete cell indices with clamping."""
    return disc.decode_cells(bits).tolist()


def min_switch_diagram(
    bdd: _bdd.BDD,
    func: _bdd.Function,
    sample_bits: dict[str, bool],
    target_label: int | None,
    disc: Discretization,
) -> tuple[float, list[str], dict[str, bool]]:
    """Algorithm MinSwitch(G, x_hat, l): shortest weighted path from root to terminal 1."""
    if func == bdd.false:
        return math.inf, [], {}

    target_var = win_flag_name(target_label) if target_label is not None else None
    cell_vars = set(disc.all_bit_names)

    # Memo stores: node -> (dist, best_bit_val, next_node)
    memo: dict[int, tuple[float, int, int]] = {}

    def dist(u: int) -> float:
        if u == -1:
            return math.inf
        if u == 1:
            return 0.0
        if u in memo:
            return memo[u][0]

        v = abs(u)
        level, lo_raw, hi_raw = bdd._bdd._succ[v]
        var = bdd._bdd.var_at_level(level)
        lo = -lo_raw if u < 0 else lo_raw
        hi = -hi_raw if u < 0 else hi_raw

        # Edge weights w(u, b)
        if var in cell_vars:
            x_hat = sample_bits.get(var, False)
            w0 = 1.0 if x_hat else 0.0
            w1 = 0.0 if x_hat else 1.0
        elif target_var is not None and var == target_var:
            w0 = math.inf   # Force target win flag = 1
            w1 = 0.0
        elif target_var is not None and var.startswith("win_"):
            w0 = 0.0
            w1 = math.inf   # Forbid rival win flag = 1
        else:
            w0 = 0.0
            w1 = 0.0

        d0 = w0 + dist(lo)
        d1 = w1 + dist(hi)

        if d0 <= d1:
            memo[u] = (d0, 0, lo)
            return d0
        else:
            memo[u] = (d1, 1, hi)
            return d1

    best_dist = dist(func.node)
    if math.isinf(best_dist):
        return math.inf, [], {}

    # Trace back optimal path from root to terminal 1
    flipped_bits: list[str] = []
    nearest_model = dict(sample_bits)

    curr = func.node
    while abs(curr) > 1:
        v = abs(curr)
        level, lo_raw, hi_raw = bdd._bdd._succ[v]
        var = bdd._bdd.var_at_level(level)
        _, choice, next_u = memo[curr]

        if var in cell_vars:
            assigned = bool(choice)
            nearest_model[var] = assigned
            if assigned != sample_bits.get(var, False):
                flipped_bits.append(var)

        curr = next_u

    return best_dist, flipped_bits, nearest_model


def min_switch_frontier(
    bdd: _bdd.BDD,
    leaves: list[LeafNode],
    sample_bits: dict[str, bool],
    target_label: int,
    disc: Discretization,
) -> tuple[float, list[str], dict[str, bool]]:
    """Compute rob_l(x_hat) across all frontier leaves."""
    best_total_dist = math.inf
    best_flipped: list[str] = []
    best_model: dict[str, bool] = {}

    for leaf in leaves:
        # Distance to enter leaf cube rho: sum_{x in rho} [rho(x) != x_hat]
        entry_dist = 0.0
        cube_flips: list[str] = []
        for var, val in leaf.rho.items():
            if val != sample_bits.get(var, False):
                entry_dist += 1.0
                cube_flips.append(var)

        if entry_dist >= best_total_dist:
            continue

        leaf_dist, leaf_flips, leaf_model = min_switch_diagram(
            bdd, leaf.acc, sample_bits, target_label, disc
        )
        total_dist = entry_dist + leaf_dist

        if total_dist < best_total_dist:
            best_total_dist = total_dist
            # Union of flips from cube entry and leaf interior
            combined_flips = list(dict.fromkeys(cube_flips + leaf_flips))
            best_flipped = combined_flips
            # Combine model
            model = dict(leaf_model)
            model.update(leaf.rho)
            best_model = model

    return best_total_dist, best_flipped, best_model


def compute_sample_robustness(
    bdd: _bdd.BDD,
    representation: dict[int, _bdd.Function] | list[LeafNode],
    sample: Sequence[float] | np.ndarray,
    disc: Discretization,
    n_classes: int,
) -> dict:
    """Compute targeted and untargeted label robustness for a continuous sample."""
    sample_bits = sample_to_bits(sample, disc)

    targeted_radii: dict[int, float] = {}
    flips_per_label: dict[int, list[str]] = {}
    features_per_label: dict[int, list[int]] = {}
    nearest_models: dict[int, dict[str, bool]] = {}

    is_frontier = isinstance(representation, list)

    for l in range(n_classes):
        if is_frontier:
            d, flips, model = min_switch_frontier(
                bdd, representation, sample_bits, target_label=l, disc=disc
            )
        else:
            win_l = representation[l]
            d, flips, model = min_switch_diagram(
                bdd, win_l, sample_bits, target_label=None, disc=disc
            )

        targeted_radii[l] = d
        flips_per_label[l] = flips
        nearest_models[l] = model

        # Map flipped bit names to feature indices
        owner = disc.bit_owner()
        touched_features: set[int] = set()
        for bit_name in flips:
            if bit_name in owner:
                touched_features.add(owner[bit_name][0])
        features_per_label[l] = sorted(touched_features)

    # Certified class is the label with radius 0 (if any)
    zero_labels = [l for l, r in targeted_radii.items() if r == 0.0]
    certified_class = zero_labels[0] if zero_labels else min(targeted_radii, key=targeted_radii.get)

    rival_labels = [l for l in range(n_classes) if l != certified_class]
    if rival_labels:
        untargeted_radius = min(targeted_radii[l] for l in rival_labels)
        nearest_rival = min(rival_labels, key=lambda l: targeted_radii[l])
    else:
        untargeted_radius = math.inf
        nearest_rival = None

    return {
        "sample_bits": sample_bits,
        "certified_class": certified_class,
        "targeted_radii": targeted_radii,
        "untargeted_radius": untargeted_radius,
        "nearest_rival": nearest_rival,
        "flips_per_label": flips_per_label,
        "features_per_label": features_per_label,
        "nearest_models": nearest_models,
    }
