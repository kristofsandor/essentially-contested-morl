"""Exact (theoretical) Pareto fronts of task vs help, per interpretation, for reach_goal.

The firefighters version (``scripts/firefighters_pareto_fronts.py``) can read a
ready-made tabular model (``transition_matrix``, ``reward_matrix_per_interp``,
``encrypt/translate/is_done``) straight off the env. ``ECCReachGoalEnv`` has none
of that: it is a plain gym env whose state is (agent position, helped-subset).
So here we *build* the tabular model on the fly by enumerating the reachable
states and reading the reward of every (state, action) straight from
``env.step``. This keeps the reward logic identical to training and needs no
duplicated transition code. It only works while the reachable set is small, so
run it on a small, deterministic layout (small grid, few humans, ``layout_seed``
set). That is the analogue of firefighters using a small tabular MDP.

Once the model is built we run finite-horizon Pareto value iteration (backward
induction, gamma = 1 because the episode is finite and terminates at the goal),
one sweep set per interpretation on that interpretation's (task, help) reward
slice, and read the front at the reset state. The hull subroutines are reused
from ``use_cases.firefighters_use_case.pareto_front``.

Why two age modes matter
------------------------
With ``age_mode="random"`` the utilitarian value of a human is independent of how
far it sits from the start->goal path, so maximising rescue count (deontological)
and maximising age-sum (utilitarian) tend to pick almost the same route: the two
fronts nearly coincide and there is little for an ECC agent to separate. With
``age_mode="depth"`` the highest-value humans are the deepest, hence the most
expensive to reach, so the utilitarian reading pays detours the deontological
reading refuses. That is the same opportunity cost ``ContestedFireFightersEnvMO``
manufactures with time pressure, and it is what pulls the two fronts apart while
each stays a clean, convex, learnable (task, help) trade-off.

Proximity shaping is excluded by construction: we build the env with
``proximity_reward=0`` so the front axes are exactly (task return, help return).

Run:
  python -m scripts.reach_goal_pareto_fronts --age-mode depth --grid-size 5 --num-humans 5
  python -m scripts.reach_goal_pareto_fronts --age-mode random --grid-size 5 --num-humans 5
"""

import argparse

import numpy as np

from env.reach_goal_ecc import ECCReachGoalEnv
from use_cases.firefighters_use_case.pareto_front import get_hull

# Reward is the flattened 2x2 matrix [task, deont_help, task, util_help].
# Interpretation i occupies columns [2*i, 2*i + 1] = (task, help).
INTERP_LABELS = ["deontological", "utilitarian"]


def _state_key(agent_pos, helped):
    return (int(agent_pos[0]), int(agent_pos[1]), tuple(int(h) for h in helped))


def _is_goal(env, agent_pos):
    return list(agent_pos) == list(env.goal_pos)


def build_model(env, max_states=200_000):
    """Enumerate reachable (pos, helped) states and read step() for each (s, a).

    Returns (states, trans, s0) where trans[s][a] = (reward_flat, next_state, terminated).
    The env layout must be deterministic (layout_seed set) for this to be exact.
    """
    env.reset()
    s0 = _state_key(env.agent_pos, env.helped)
    n_actions = env.action_space.n

    trans = {}
    seen = {s0}
    stack = [s0]
    while stack:
        s = stack.pop()
        r0, c0, helped = s
        if _is_goal(env, (r0, c0)):
            trans[s] = {}  # terminal: no outgoing transitions needed
            continue
        trans[s] = {}
        for a in range(n_actions):
            # Drive the env from this exact state, then read the true reward.
            env.agent_pos = [r0, c0]
            env.prev_pos = [r0, c0]
            env.helped = np.array(helped, dtype=bool)
            _, reward, terminated, _, _ = env.step(a)
            ns = _state_key(env.agent_pos, env.helped)
            trans[s][a] = (np.asarray(reward, dtype=float), ns, bool(terminated))
            if ns not in seen:
                seen.add(ns)
                stack.append(ns)
        if len(seen) > max_states:
            raise RuntimeError(
                f"reachable set exceeded {max_states} states; use a smaller "
                f"grid_size / num_humans for the exact front."
            )
    return sorted(seen), trans, s0


def finite_horizon_fronts(states, trans, s0, env, horizon, pareto=True):
    """Exact finite-horizon Pareto fronts via backward induction, per interpretation.

    V_t(s) = ND over a of { r_i(s, a) + V_{t-1}(next(s, a)) }, gamma = 1.
    A step that terminates (reaches the goal) contributes zero future return; the
    terminal reward is already inside r(s, a).
    """
    zero = np.zeros((1, 2))
    n_interp = len(INTERP_LABELS)
    fronts = {}
    for i, label in enumerate(INTERP_LABELS):
        cols = [2 * i, 2 * i + 1]  # (task, help) slice for this interpretation
        V_next = {s: zero for s in states}
        for _ in range(horizon):
            V_t = {}
            for s in states:
                if not trans[s]:  # terminal / goal state
                    V_t[s] = zero
                    continue
                pts = []
                for a, (reward, ns, terminated) in trans[s].items():
                    ri = reward[cols][None, :]
                    fut = zero if terminated else V_next.get(ns, zero)
                    pts.append(ri + fut)
                V_t[s] = get_hull(np.unique(np.vstack(pts), axis=0), pareto=pareto)
            V_next = V_t
        f = get_hull(np.asarray(V_next[s0]), pareto=pareto)
        f = f[np.lexsort((f[:, 1], f[:, 0]))]
        fronts[label] = f
    return fronts


def normalize_fronts(fronts, mode="none"):
    """Rescale fronts to [0, 1] so trade-off shapes are comparable across interps."""
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


def plot_fronts(fronts, title, save_path, normalized=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.5, 6))
    for label, f in fronts.items():
        plt.plot(f[:, 0], f[:, 1], "-", alpha=0.4, lw=1)
        plt.scatter(f[:, 0], f[:, 1], s=26, label=f"{label} ({len(f)})")
    if normalized:
        plt.xlabel("normalized task return")
        plt.ylabel("normalized help return")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
    else:
        plt.xlabel("task return")
        plt.ylabel("help return")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age-mode", choices=["random", "depth"], default="depth")
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--num-humans", type=int, default=5)
    ap.add_argument("--layout-seed", type=int, default=0)
    ap.add_argument("--step-penalty", type=float, default=0.1)
    ap.add_argument("--terminal-reward", type=float, default=1.0)
    ap.add_argument("--help-reward", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=None,
                    help="backward-induction depth; defaults to a value large enough "
                         "to rescue everyone and reach the goal")
    ap.add_argument("--convex", action="store_true",
                    help="convex hull (CCS) instead of the full Pareto front")
    ap.add_argument("--normalize", choices=["none", "own", "shared"], default="none")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    env = ECCReachGoalEnv(
        grid_size=args.grid_size,
        num_humans=args.num_humans,
        step_penalty=args.step_penalty,
        terminal_reward=args.terminal_reward,
        help_reward=args.help_reward,
        proximity_reward=0.0,  # exclude shaping: axes are exactly (task, help)
        obs_as_grid=False,
        layout_seed=args.layout_seed,
        age_mode=args.age_mode,
    )

    # Generous horizon: every human can be rescued and the goal reached.
    horizon = args.horizon or (4 * args.grid_size * (args.num_humans + 1))

    print(f"age_mode={args.age_mode}  grid={args.grid_size}  humans={args.num_humans}  "
          f"layout_seed={args.layout_seed}")
    print("human (row, col, age):")
    for pos, age in zip(env.human_positions, env.human_ages):
        print(f"  ({pos[0]}, {pos[1]})  age={age:.2f}")

    states, trans, s0 = build_model(env)
    print(f"reachable states: {len(states)}  horizon: {horizon}")

    fronts = finite_horizon_fronts(states, trans, s0, env, horizon, pareto=not args.convex)
    for label, f in fronts.items():
        print(f"\n{label}: {len(f)} points (task, help)")
        print(np.round(f, 3))

    if not args.no_plot:
        plot_data = normalize_fronts(fronts, args.normalize)
        kind = "convex" if args.convex else "pareto"
        suffix = f"_{args.normalize}" if args.normalize != "none" else ""
        save_path = f"reach_goal_{kind}_fronts_{args.age_mode}{suffix}.png"
        title = (f"Reach-goal theoretical {kind} fronts "
                 f"(age_mode={args.age_mode}, grid={args.grid_size}, "
                 f"humans={args.num_humans})")
        if args.normalize != "none":
            title = title.replace("fronts", "normalized fronts")
        plot_fronts(plot_data, title, save_path, normalized=args.normalize != "none")


if __name__ == "__main__":
    main()
