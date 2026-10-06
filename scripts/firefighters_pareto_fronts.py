"""Exact (theoretical) Pareto fronts of professionalism vs proximity, per interpretation.

Because each interpretation of ``ECCFireFightersEnvMO`` is a finite, deterministic,
tabular two-objective MDP with a known reward matrix, its Pareto front of
(professionalism, proximity) returns is computable exactly by Pareto value
iteration. This reuses the repo's hull subroutines (``translate_hull``,
``get_hull``) and runs one value-iteration sweep per interpretation on that
interpretation's reward slice (``env.reward_matrix_per_interp[:, :, i, :]``),
reading the front at the canonical initial state [0, 3, 4, 0, 0, 3] (id 323).

Notes:
  * The base env uses DISCOUNT = 1.0, which does not converge for non-terminating
    loops. Use gamma < 1 here. gamma=0.95 converges in ~130 sweeps; for gamma=0.99
    raise ``iters`` to ~400.
  * ``pareto=True`` returns the full (possibly non-convex) Pareto front. Set
    ``pareto=False`` for the convex hull (the convex coverage set that linear
    scalarisation / Envelope / GPI-LS can recover).

Run:  python -m scripts.firefighters_pareto_fronts --preset all_five --gamma 0.95 --iters 130
"""

import argparse

import numpy as np

from env.firefighters_ecc import (
    ContestedFireFightersEnvMO,
    ECCFireFightersEnvMO,
    INTERPRETATION_PRESETS,
)
from use_cases.firefighters_use_case.pareto_front import get_hull, translate_hull

INITIAL_STATE = np.array([0, 3, 4, 0, 0, 3])


def _next_state(env, s, a):
    """Deterministic successor of (s, a) from the tabular transition matrix."""
    return int(np.argmax(env.transition_matrix[s, a]))


def _is_done(env, s):
    return bool(env.real_env.is_done(env.real_env.translate(s)))


def reachable_states(env, s0):
    """Forward-reachable closure from s0; the front at s0 only depends on these."""
    seen, stack = {s0}, [s0]
    while stack:
        s = stack.pop()
        if _is_done(env, s):
            continue
        for a in range(env.action_space.n):
            ns = _next_state(env, s, a)
            if ns not in seen:
                seen.add(ns)
                stack.append(ns)
    return sorted(seen)


def pareto_value_iteration(env, reward, gamma, iters, states, pareto=True):
    """Pareto VI for one (S, A, 2) reward matrix. Returns {state: front}.

    V(s) = ND over actions a of { reward[s, a] + gamma * V(next(s, a)) }, with
    terminal states contributing the empty hull (future return 0).
    """
    V = {s: np.zeros((0, 2)) for s in states}
    for _ in range(iters):
        new_V = {}
        for s in states:
            if _is_done(env, s):
                new_V[s] = np.zeros((0, 2))
                continue
            pts = []
            for a in range(env.action_space.n):
                hull = V[_next_state(env, s, a)]
                # translate_hull(point, gamma, hull) = gamma * hull + point,
                # or [point] when hull is empty (terminal successor).
                sa = translate_hull(np.asarray(reward[s, a], dtype=float), gamma, hull)
                pts.extend(np.asarray(sa).reshape(-1, 2))
            new_V[s] = get_hull(np.unique(np.asarray(pts), axis=0), pareto=pareto)
        V = new_V
    return V


def fronts_per_interpretation(env, gamma=0.95, iters=130, pareto=True):
    """Compute the initial-state Pareto front for every interpretation in env."""
    s0 = int(env.real_env.encrypt(INITIAL_STATE))
    states = reachable_states(env, s0)
    fronts = {}
    for i, label in enumerate(env.interpretation_labels):
        reward = env.reward_matrix_per_interp[:, :, i, :]
        V = pareto_value_iteration(env, reward, gamma, iters, states, pareto)
        f = get_hull(np.asarray(V[s0]), pareto=pareto)
        f = f[np.lexsort((f[:, 1], f[:, 0]))]
        fronts[label] = f
    return fronts, s0


def finite_horizon_fronts(env, horizon, pareto=True):
    """Exact finite-horizon Pareto fronts via backward induction.

    Required for the contested env, where time pressure makes the horizon matter
    and infinite-horizon value iteration would not terminate. Computes, for each
    interpretation, the front of (professionalism, proximity) returns achievable
    in ``horizon`` steps from the initial state.
    """
    s0 = int(env.real_env.encrypt(INITIAL_STATE))
    states = reachable_states(env, s0)
    zero = np.zeros((1, 2))
    fronts = {}
    for i, label in enumerate(env.interpretation_labels):
        reward = env.reward_matrix_per_interp[:, :, i, :]
        V_next = {s: zero for s in states}  # 0 future reward at the horizon
        for _ in range(horizon):
            V_t = {}
            for s in states:
                if _is_done(env, s):
                    V_t[s] = zero
                    continue
                pts = []
                for a in range(env.action_space.n):
                    fut = V_next.get(_next_state(env, s, a), zero)
                    pts.append(np.asarray(reward[s, a], dtype=float)[None, :] + fut)
                V_t[s] = get_hull(np.unique(np.vstack(pts), axis=0), pareto=pareto)
            V_next = V_t
        f = get_hull(np.asarray(V_next[s0]), pareto=pareto)
        f = f[np.lexsort((f[:, 1], f[:, 0]))]
        fronts[label] = f
    return fronts, s0


def _scalarized_cross_return(env, R, target, w, horizon, states, s0):
    """4-vector return of the policy that maximizes ``w . (interp `target` return)``.

    Linear scalarization is what Envelope / GPI-PD actually optimize, so sweeping
    ``w`` over the 2-simplex traces the convex coverage set (CCS) they recover. This
    runs scalar backward induction on the objective ``w . R[:, :, target, :]`` while
    carrying, as bookkeeping, the *full* per-interpretation return of the argmax
    policy. That bookkeeping return is what lets us score one interpretation's
    optimal policies under the *other* interpretation's reward for free.

    R      : (S, A, num_interps, 2) reward tensor (``reward_matrix_per_interp``).
    target : which interpretation the policy is optimized for.
    Returns the concatenated return [i0_pf, i0_px, i1_pf, i1_px, ...] from ``s0``.
    """
    n_book = R.shape[2] * R.shape[3]  # num_interps * 2
    V = {s: 0.0 for s in states}
    B = {s: np.zeros(n_book) for s in states}
    for _ in range(horizon):
        V_t, B_t = {}, {}
        for s in states:
            if _is_done(env, s):
                V_t[s], B_t[s] = 0.0, np.zeros(n_book)
                continue
            best_val, best_b = -np.inf, None
            for a in range(env.action_space.n):
                ns = _next_state(env, s, a)
                val = float(np.dot(w, R[s, a, target])) + V[ns]
                if val > best_val:
                    best_val = val
                    best_b = np.asarray(R[s, a], dtype=float).reshape(-1) + B[ns]
            V_t[s], B_t[s] = best_val, best_b
        V, B = V_t, B_t
    return B[s0]


def theoretical_cross_fronts(env, horizon, num_weights=100):
    """Cross-evaluated theoretical fronts for every interpretation of ``env``.

    For each interpretation ``target``, sweeps the objective weight ``w`` over the
    (professionalism, proximity) 2-simplex, computes the exact ``w``-optimal policy
    on the contested env, and records its return under *all* interpretations. So
    ``cross[label]`` is an ``(N, num_interps * 2)`` array: for the policies optimal
    for ``label``, columns ``2*e : 2*e+2`` are their return under interpretation
    ``e``. Plotting column-block ``e`` for every ``label`` on subplot ``e`` gives,
    per reward structure, both its own optimal front and the fronts the other
    interpretations' optimal policies achieve under it.
    """
    s0 = int(env.real_env.encrypt(INITIAL_STATE))
    states = reachable_states(env, s0)
    R = env.reward_matrix_per_interp  # (S, A, num_interps, 2)
    weights = [np.array([1.0 - t, t]) for t in np.linspace(0.0, 1.0, num_weights)]
    cross = {}
    for target, label in enumerate(env.interpretation_labels):
        pts = np.array(
            [_scalarized_cross_return(env, R, target, w, horizon, states, s0)
             for w in weights]
        )
        cross[label] = np.unique(pts, axis=0)
    return cross, s0


def _pareto_front(points):
    """Non-dominated subset (both objectives maximized), sorted by objective 0."""
    from morl_baselines.common.pareto import get_non_dominated

    nd = get_non_dominated({tuple(p) for p in np.asarray(points, dtype=float)})
    return np.array(sorted(nd, key=lambda p: p[0]))


def plot_contested_theoretical(
    cross,
    interpretation_labels,
    title="Theoretical contested Pareto front",
    normalize="shared",
    share_limits=False,
    save_path="ff_theoretical_contested.png",
    figsize=(16, 7),
    dpi=150,
):
    """One subplot per reward structure; every interpretation's optimal front on each.

    Mirrors the trained-agent figure (``plot_pareto_fronts`` in eval_ff.ipynb): subplot
    ``e`` shows, scored under interpretation ``e``'s reward, the Pareto front of every
    interpretation's ``w``-optimal policies (its own = the true theoretical front, the
    others = what committing to a rival interpretation gets you here). HV/EUM per front
    reported exactly as in that figure. ``normalize='shared'`` rescales each subplot to
    [0, 1] by the across-front min/max (recommended; matches the *_norm figures).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from morl_baselines.common.performance_indicators import (
        expected_utility,
        hypervolume,
    )
    from morl_baselines.common.weights import equally_spaced_weights

    assert normalize in ("none", "shared"), normalize
    weights_set = equally_spaced_weights(2, 100)  # for the Expected Utility Metric
    policy_labels = list(cross.keys())
    # rescue = red, fire = green, to match the trained-agent figure's convention.
    palette = {"rescue_minded": "tab:red", "fire_minded": "tab:green"}
    fallback = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = {pl: palette.get(pl) or next(fallback) for pl in policy_labels}
    prefix = "normalized " if normalize != "none" else ""

    # common raw-return limits across every subplot (only meaningful un-normalized)
    common_xlim = common_ylim = None
    if share_limits and normalize == "none":
        blocks = {
            (e, pl): cross[pl][:, 2 * e:2 * e + 2]
            for e in range(len(interpretation_labels))
            for pl in policy_labels
        }
        common_xlim, common_ylim = shared_limits(blocks)

    fig, axes = plt.subplots(1, len(interpretation_labels), figsize=figsize, squeeze=False)
    for ax, (e, interp) in zip(axes[0], enumerate(interpretation_labels)):
        # each policy set's return under THIS reward structure (columns 2e:2e+2)
        loaded = {pl: cross[pl][:, 2 * e:2 * e + 2] for pl in policy_labels}
        all_pts = np.vstack(list(loaded.values()))
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        ref_point = lo - 0.05 * span

        for pl in policy_labels:
            pts = loaded[pl]
            pf = _pareto_front(pts)
            eum = expected_utility((pf - lo) / span, weights_set)  # shared-normalized
            if normalize == "shared":
                d_pts, d_pf, d_ref = (pts - lo) / span, (pf - lo) / span, (ref_point - lo) / span
            else:
                d_pts, d_pf, d_ref = pts, pf, ref_point
            hv = hypervolume(d_ref, d_pf)
            color = colors[pl]
            ax.scatter(d_pts[:, 0], d_pts[:, 1], s=20, alpha=0.3, color=color)
            ax.plot(d_pf[:, 0], d_pf[:, 1], marker="o", color=color, linewidth=2.5,
                    label=f"{pl}  (HV={hv:.2f}, EUM={eum:.3f})")

        rp = ref_point if normalize == "none" else (ref_point - lo) / span
        ax.scatter(*rp, marker="x", color="black", s=60, label="ref point")
        ax.set_title(f"{title} — {interp}")
        ax.set_xlabel(prefix + "professionalism")
        ax.set_ylabel(prefix + "proximity")
        if normalize != "none":
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
        elif common_xlim is not None:
            ax.set_xlim(common_xlim)
            ax.set_ylim(common_ylim)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"saved {save_path}")


def normalize_fronts(fronts, mode="own"):
    """Rescale fronts to a common [0, 1] range so trade-off *shapes* are comparable.

    Each interpretation has its own reward magnitudes, so the raw fronts sit at
    different heights and the overlay is dominated by scale. Normalising removes
    that and exposes the shape of each trade-off. It does not change the
    underlying conflict between interpretations, only how the fronts are drawn.

    mode='own'    : each front scaled by its own per-objective ideal (max) and
                    nadir (min); every front then fills [0, 1]^2, isolating shape.
    mode='shared' : all fronts scaled by a single global per-objective min/max,
                    preserving their relative positions on a 0-1 scale.
    """
    if mode in (None, "none"):
        return fronts
    if mode == "shared":
        allpts = np.vstack(list(fronts.values()))
        lo, hi = allpts.min(axis=0), allpts.max(axis=0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        return {k: (v - lo) / rng for k, v in fronts.items()}
    if mode == "own":
        out = {}
        for k, v in fronts.items():
            lo, hi = v.min(axis=0), v.max(axis=0)
            rng = np.where(hi > lo, hi - lo, 1.0)
            out[k] = (v - lo) / rng
        return out
    raise ValueError(f"unknown normalize mode {mode!r}")


def shared_limits(fronts=None):
    """Fixed common (xlim, ylim) for drawing all fronts on one raw-return scale.

    An alternative to normalizing: keep the true return magnitudes but draw every
    front on the same axes, chosen to be "good for both" interpretations.
    """
    return (0.5, 5.0), (-0.5, 3.5)


def plot_fronts(fronts, title, save_path="ecc_pareto_fronts.png",
                normalized=False, xlim=None, ylim=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.5, 6))
    for label, f in fronts.items():
        plt.plot(f[:, 0], f[:, 1], "-", alpha=0.4, lw=2.5, )
        plt.scatter(f[:, 0], f[:, 1], s=22, label=f"{label} ({len(f)})")
    if normalized:
        plt.xlabel("normalized professionalism")
        plt.ylabel("normalized proximity")
        plt.xlim(xlim if xlim is not None else (-0.05, 1.05))
        plt.ylim(ylim if ylim is not None else (-0.05, 1.05))
    else:
        plt.xlabel("professionalism return")
        plt.ylabel("proximity return")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        # otherwise let matplotlib autoscale to the data
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="rescue_vs_fire", choices=sorted(INTERPRETATION_PRESETS))
    ap.add_argument("--env", choices=["base", "contested"], default="contested",
                    help="contested adds time pressure so interpretations conflict")
    ap.add_argument("--horizon", type=int, default=5,
                    help="finite horizon for the contested env (backward induction)")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--iters", type=int, default=130)
    ap.add_argument("--convex", action="store_true", help="convex hull instead of full Pareto front")
    ap.add_argument("--normalize", choices=["none", "own", "shared"], default="none",
                    help="rescale fronts to [0,1] so trade-off shapes are comparable")
    ap.add_argument("--share-limits", action="store_true",
                    help="keep raw returns but draw all fronts on common x/y limits "
                         "(union bounding box, padded) chosen to fit both")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--cross", action="store_true",
                    help="two-subplot cross-evaluated figure: each reward structure's own "
                         "theoretical front plus the front the other's optimal policies reach")
    ap.add_argument("--num-weights", type=int, default=100,
                    help="linear-scalarization weights swept per interpretation (--cross)")
    args = ap.parse_args()

    if args.cross:
        # Always contested: with the base env's slack, both interpretations are near-
        # optimal for each other and the cross fronts collapse onto one another.
        env = ContestedFireFightersEnvMO.from_preset(args.preset, horizon=args.horizon)
        cross, s0 = theoretical_cross_fronts(env, args.horizon, num_weights=args.num_weights)
        print(f"preset={args.preset}  initial state id={s0}  cross-evaluated CCS")
        for label, pts in cross.items():
            print(f"\n{label}-optimal policies: {len(pts)} distinct returns "
                  f"[cols per interp: {env.interpretation_labels}]")
            print(np.round(pts, 3))
        if not args.no_plot:
            # match the reference *_norm figure (shared per-subplot rescaling) by
            # default; --normalize none plots raw returns instead.
            norm = "none" if args.normalize == "none" else "shared"
            tag = norm + ("_shared" if args.share_limits and norm == "none" else "")
            plot_contested_theoretical(
                cross,
                env.interpretation_labels,
                title="Theoretical contested Pareto front",
                normalize=norm,
                share_limits=args.share_limits,
                save_path=f"ff_theoretical_contested_{tag}.png",
            )
        return

    if args.env == "contested":
        env = ContestedFireFightersEnvMO.from_preset(args.preset, horizon=args.horizon)
        fronts, s0 = finite_horizon_fronts(env, args.horizon, pareto=not args.convex)
    else:
        env = ECCFireFightersEnvMO.from_preset(args.preset)
        fronts, s0 = fronts_per_interpretation(
            env, gamma=args.gamma, iters=args.iters, pareto=not args.convex
        )
    set_name = "convex" if args.convex else "pf"
    set_title = "CCS" if args.convex else "Pareto Front"
    print(f"preset={args.preset}  initial state id={s0}  set={set_name}")
    for label, f in fronts.items():
        print(f"\n{label}: {len(f)} points (professionalism, proximity) [{set_name}]")
        print(np.round(f, 3))

    plot_data = normalize_fronts(fronts, args.normalize)
    if not args.no_plot:
        suffix = f"_{args.normalize}" if args.normalize != "none" else ""
        if args.env == "contested":
            title = f"Contested FF {set_title} (horizon={args.horizon}, time pressure)"
            base_name = f"ecc_{set_name}_{args.preset}_contested"
        else:
            title = f"Theoretical {set_title} at initial state (gamma={args.gamma})"
            base_name = f"ecc_{set_name}_{args.preset}"
        if args.normalize != "none":
            title = f"normalized {title}"
        xlim, ylim = (shared_limits(plot_data) if args.share_limits else (None, None))
        if args.share_limits:
            suffix += "_shared"
        plot_fronts(
            plot_data,
            title,
            save_path=f"{base_name}{suffix}.png",
            normalized=args.normalize != "none",
            xlim=xlim,
            ylim=ylim,
        )


if __name__ == "__main__":
    main()
