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

from env.firefighters_ecc import ECCFireFightersEnvMO, INTERPRETATION_PRESETS
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


def plot_fronts(fronts, gamma, save_path="ecc_pareto_fronts.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.5, 6))
    for label, f in fronts.items():
        plt.plot(f[:, 0], f[:, 1], "-", alpha=0.4, lw=1)
        plt.scatter(f[:, 0], f[:, 1], s=22, label=f"{label} ({len(f)})")
    plt.xlabel("professionalism return")
    plt.ylabel("proximity return")
    plt.title(f"Theoretical Pareto fronts at initial state (gamma={gamma})")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="rescue_vs_fire", choices=sorted(INTERPRETATION_PRESETS))
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--iters", type=int, default=130)
    ap.add_argument("--convex", action="store_true", help="convex hull instead of full Pareto front")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    env = ECCFireFightersEnvMO.from_preset(args.preset)
    fronts, s0 = fronts_per_interpretation(
        env, gamma=args.gamma, iters=args.iters, pareto=not args.convex
    )
    print(f"preset={args.preset}  initial state id={s0}")
    for label, f in fronts.items():
        print(f"\n{label}: {len(f)} points (professionalism, proximity)")
        print(np.round(f, 3))
    if not args.no_plot:
        plot_fronts(fronts, args.gamma, save_path=f"ecc_pareto_{args.preset}.png")


if __name__ == "__main__":
    main()
