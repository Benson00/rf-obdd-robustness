from __future__ import annotations

import dd.autoref as _bdd
from rfbdd.forest import Forest, Tree
from rfbdd.discretize import Discretization
from rfbdd.first_tree import (
    compute_counter_width_m,
    counter_bit_name,
    win_flag_name,
    encode_path_formula,
)


def encode_geq_offset(
    bdd: _bdd.BDD,
    label_s: int,
    label_l: int,
    r_t: int,
    m: int,
):
    """Encode the offset comparator LC^s >= LC^l + R_t using formula chi_M (Definition 9)."""
    # Build carries kappa_j for 1 <= j <= M
    # kappa_j asserts: (b_{j-1}...b_0)_2 >= 2^j - (R_t mod 2^j)
    kappas: list = [bdd.false]  # kappa_0 = false
    for j in range(1, m + 1):
        mod_val = r_t % (1 << j)
        if mod_val == 0:
            kappas.append(bdd.false)
        else:
            thresh = (1 << j) - mod_val
            # Evaluate comparator of low j bits of LC^l against thresh
            # Low-to-high recurrence matching Definition 4
            psi = bdd.true
            for p in range(j):
                thresh_p = (thresh >> p) & 1
                b_p = bdd.var(counter_bit_name(label_l, p, barred=False))
                not_b = ~b_p
                psi = (not_b & psi) if thresh_p == 0 else (not_b | psi)
            # The comparator asserts <= thresh - 1, so geq threshold is its negation:
            # But directly: (b >= thresh) is not(b <= thresh - 1)
            # Alternatively, test directly bit by bit:
            # Let's build geq threshold directly:
            # geq(x, thresh) == not leq(x, thresh - 1)
            psi_leq = bdd.true
            target = thresh - 1
            for p in range(j):
                t_p = (target >> p) & 1
                b_p = bdd.var(counter_bit_name(label_l, p, barred=False))
                not_b = ~b_p
                psi_leq = (not_b & psi_leq) if t_p == 0 else (not_b | psi_leq)
            kappas.append(~psi_leq)

    # chi_j recurrence from j = 0 to M (processed from 0 up to M)
    # chi_{-1} = true
    chi = bdd.true
    for j in range(m + 1):
        r_j = (r_t >> j) & 1
        a_j = bdd.var(counter_bit_name(label_s, j, barred=False))
        b_j = bdd.var(counter_bit_name(label_l, j, barred=False))
        kappa_j = kappas[j]

        # g_hat_j = b_j XOR r_j XOR kappa_j
        # Since r_j is constant:
        b_xor_k = (b_j & ~kappa_j) | (~b_j & kappa_j)
        g_hat_j = ~b_xor_k if r_j == 1 else b_xor_k

        # chi_j = (a_j AND ~g_hat_j) OR ((a_j <-> g_hat_j) AND chi_{j-1})
        strictly_greater = a_j & ~g_hat_j
        equal = (a_j & g_hat_j) | (~a_j & ~g_hat_j)
        chi = strictly_greater | (equal & chi)

    return chi


def encode_clinch(
    bdd: _bdd.BDD,
    label_s: int,
    t_idx: int,
    forest: Forest,
    m: int,
):
    """Encode the Clinch_s^(t) predicate (Definition 9)."""
    d = forest.D
    w = forest.W
    r_t = d - 1 - t_idx

    # Syntactic short-circuit: if R_t > W - 1 (equivalently t < floor(D/2)), Clinch is false
    if r_t > w - 1:
        return bdd.false

    clinch = bdd.true
    for l in range(forest.n_classes):
        if l != label_s:
            comp = encode_geq_offset(bdd, label_s=label_s, label_l=l, r_t=r_t, m=m)
            clinch &= comp
    return clinch


def encode_counter_copy(
    bdd: _bdd.BDD,
    label: int,
    m: int,
):
    """Encode equality between incoming and outgoing counters: (LC_bar^l = LC^l)."""
    eq = bdd.true
    for j in range(m + 1):
        curr_b = bdd.var(counter_bit_name(label, j, barred=False))
        next_b = bdd.var(counter_bit_name(label, j, barred=True))
        # curr_b <-> next_b
        eq &= ((curr_b & next_b) | (~curr_b & ~next_b))
    return eq


def encode_counter_increment(
    bdd: _bdd.BDD,
    label: int,
    m: int,
):
    """Encode increment: (LC_bar^label = LC^label + 1)."""
    eq = bdd.true
    carry = bdd.true  # incoming carry +1 at bit 0
    for j in range(m + 1):
        curr_b = bdd.var(counter_bit_name(label, j, barred=False))
        next_b = bdd.var(counter_bit_name(label, j, barred=True))
        sum_bit = (curr_b & ~carry) | (~curr_b & carry)
        eq &= ((next_b & sum_bit) | (~next_b & ~sum_bit))
        carry = curr_b & carry
    return eq


def encode_hold(
    bdd: _bdd.BDD,
    n_classes: int,
):
    """Encode Hold condition latching win flags and leaving counters free."""
    hold = bdd.true
    for l in range(n_classes):
        curr_w = bdd.var(win_flag_name(l, barred=False))
        next_w = bdd.var(win_flag_name(l, barred=True))
        hold &= ((curr_w & next_w) | (~curr_w & ~next_w))
    return hold


def encode_subsequent_tree(
    bdd: _bdd.BDD,
    forest: Forest,
    disc: Discretization,
    t_idx: int,
    rho: dict[str, bool] | None = None,
):
    """Encode formula Phi_t for tree T_t (1 <= t <= D - 1) (Definition 10)."""
    tree = forest.trees[t_idx]
    taus_t = disc.taus[t_idx]
    n_classes = forest.n_classes
    m = compute_counter_width_m(forest.D)

    # Condition won = OR_l win_l
    won = bdd.false
    for l in range(n_classes):
        won |= bdd.var(win_flag_name(l, barred=False))

    # Hold branch: latches win flags, leaves counters unconstrained
    hold = encode_hold(bdd, n_classes)

    # Step_t branch: disjunction over leaves of T_t
    step_t = bdd.false
    for steps, leaf_label in tree.paths():
        pi_b = encode_path_formula(bdd, disc, steps, taus_t, rho=rho)

        clinch_s = encode_clinch(bdd, label_s=leaf_label, t_idx=t_idx, forest=forest, m=m)

        # Wins branch: win_bar_s = true, others false, counters free
        wins = bdd.var(win_flag_name(leaf_label, barred=True))
        for other_l in range(n_classes):
            if other_l != leaf_label:
                wins &= ~bdd.var(win_flag_name(other_l, barred=True))

        # Incs branch: increment s, copy rivals, all win_bar flags false
        incs = encode_counter_increment(bdd, label=leaf_label, m=m)
        for other_l in range(n_classes):
            if other_l != leaf_label:
                incs &= encode_counter_copy(bdd, label=other_l, m=m)
        for l in range(n_classes):
            incs &= ~bdd.var(win_flag_name(l, barred=True))

        choice = (clinch_s & wins) | (~clinch_s & incs)
        step_t |= (pi_b & choice)

    return (won & hold) | (~won & step_t)

