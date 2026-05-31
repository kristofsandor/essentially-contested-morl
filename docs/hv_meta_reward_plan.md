# Plan: Hypervolume-Guided Weight Adaptation for `ecc_envelope_weight_adapt.py`

Source: Lu et al., *Learning to Optimize Multi-Objective Alignment Through Dynamic Reward
Weighting* (arXiv:2509.11452v2), §4 + Algorithm 1.

## 1. What the paper does

The paper's **hypervolume-guided weight adaptation** keeps the human-specified objective
weights `w` fixed and instead amplifies the *reward signal* with a meta-level reward that
fires whenever a new checkpoint expands the Pareto front:

- Maintain a performance buffer `B` = the current Pareto set (validation returns).
- Each training step `t`:
  1. roll out, compute per-objective reward vectors `r_i` and scalar reward `r_i = wᵀr_i`;
  2. shape the reward: `r̃_i = r_pareto · r_i` (Alg. 1, line 12);
  3. update the policy on the shaped reward;
  4. evaluate, compute `ΔHV(r_θt, B)` = hypervolume contribution of the new return;
  5. set `r_pareto = 0.5 + 1.5·tanh(ΔHV(r_θt, B))` (Eq. 1);
  6. if `ΔHV > 0`, add `r_θt` to `B`.
- `r_pareto` is initialised to `1` ("no hypervolume contribution yet", line 5), and the value
  used at step `t` is the one computed at the end of step `t-1`. So the shaping reward is a
  slowly-varying multiplicative signal in `[0.5, 2.0]` (since `tanh ∈ (-1,1)`).

Intuition: episodes/checkpoints that push the front outward get their gradient amplified
(up to ~2×); episodes that don't get damped (down to ~0.5×). This redirects learning effort
toward the frontier without touching the user's preference weights `w`.

## 2. Translating to Envelope Q-learning (this file)

The paper is policy-gradient (GRPO/REINFORCE) with a "training step = batch of rollouts +
eval" loop. This file is **off-policy Envelope Q-learning** with a per-interpretation network
ensemble. The mapping:

| Paper concept | This codebase |
|---|---|
| training step `t` | one completed **episode** |
| validation return `r_θt` | the episode's MO return (already tracked in `_episode_mo_returns_per_interp`) |
| Pareto buffer `B` | a non-dominated set of past returns (new state) |
| reward shaping `r̃_i = r_pareto·r_i` | scale `vec_reward` **when added to the replay buffer** |
| `r_pareto` from step `t-1` | running `self._meta_reward`, updated at each episode end |

Why scale at buffer-insert time (not at `update()` time): it stamps each transition with the
`r_pareto` that was in effect when it was collected — faithful to Algorithm 1, where the
shaped reward of a step uses the previous step's meta-reward. Scaling at update time would
retroactively re-weight all stored transitions with the current value, which is not what the
paper does and interacts badly with the replay buffer + PER priorities.

### Per-interpretation handling
The agent already maintains one Pareto front per ethical interpretation
(`_episode_mo_returns_per_interp`, `_marginal_hv_sum`). The natural fit is a **per-interp
meta-reward**: compute `ΔHV` against interpretation `i`'s own buffer `B_i`, get `r_pareto,i`,
and scale row `i` of the flattened reward matrix. A `hv_meta_per_interp=False` mode collapses
to a single global buffer in agent-objective space (closer to the paper's single-`w` setting),
scaling all rows by one scalar.

### Relationship to existing UCB code
This is **additive and orthogonal** to `ucb_best_weight_and_interp`. UCB *selects which weight
to train under* each episode; the meta-reward *amplifies the reward magnitude* for the chosen
weight. Both can be on at once, and the new mechanism is gated behind `use_hv_meta` (default
`False`) so existing runs are unchanged.

## 3. Concrete changes to `agent/ecc_envelope_weight_adapt.py`

1. **`__init__` signature** — add:
   `use_hv_meta: bool = False`, `hv_meta_base: float = 0.5`, `hv_meta_scale: float = 1.5`,
   `hv_meta_per_interp: bool = True`.
2. **`__init__` body** — store them; init state:
   - `self._meta_reward = np.ones(num_interps, dtype=np.float32)` (per-interp) — the running
     `r_pareto`, starts at 1 per the paper;
   - `self._hv_meta_buffer = [[] for _ in range(num_interps)]` — per-interp Pareto set `B_i`
     (list of return tuples);
   - `self._hv_meta_buffer_global = []` for the global mode.
3. **`get_config`** — surface the four new params for wandb.
4. **New helper `_update_hv_meta_reward(per_interp_return, agent_return)`** — for each interp
   (or global): `ΔHV = HV(B ∪ {r}) − HV(B)`, `r_pareto = base + scale·tanh(ΔHV)`, and if
   `ΔHV > 0` insert `r` and re-filter via `get_non_dominated`. Updates `self._meta_reward` and
   returns the meta values + per-interp `ΔHV` for logging. Reuses already-imported
   `hypervolume` and `get_non_dominated`; `ref_point` is the existing HV reference.
5. **`train()` loop**:
   - At the buffer-add line, when `use_hv_meta`, reshape `vec_reward` to
     `[num_interps, net_reward_dim]`, multiply row `i` by `self._meta_reward[i]` (or the global
     scalar), flatten, and store the scaled reward. The unscaled reward is still used for
     episode-statistics/HV bookkeeping.
   - At episode end (where `_episode_mo_returns_per_interp` is appended), call
     `_update_hv_meta_reward(...)` using the **unscaled** env return, and log
     `hv_meta/r_pareto_*` and `hv_meta/delta_hv_*`.

## 4. Notes / caveats

- The shaping multiplies the per-interp objective vector, so the TD targets and PER priorities
  scale accordingly — this is intended (it is the gradient amplification mechanism).
- Returns can be negative depending on the env; `tanh(ΔHV)` keeps the multiplier bounded in
  `[base−scale, base+scale] = [-1.0, 2.0]`. With default `(0.5, 1.5)` a strongly negative
  `ΔHV` could yield a slightly negative multiplier; if undesirable for a given env, clamp the
  multiplier to `≥ 0` (left as a config-free guard in the helper). The paper reports `ΔHV ≥ 0`
  by construction (contribution is non-negative), which holds here too since we only ever
  compare against the current front, so the multiplier stays in `[0.5, 2.0]` in practice.
- Buffers persist across episodes within a `train()` call; they grow only with non-dominated
  points so they stay small.

## 5. Verification

- `python -m py_compile` the edited file.
- Unit-style numeric check of `_update_hv_meta_reward`: feed a sequence of dominating /
  dominated returns and assert the meta-reward rises toward `2.0` on front expansion and falls
  toward `0.5` otherwise, and that `B` only retains non-dominated points.
