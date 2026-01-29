"""
Policy comparison entrypoint.

Runs multiple policies from `policies/` on the same environment, collects metrics,
and saves a comparison figure to an image (no interactive window required).

Example:
  python main.py --env cartpole --policies ppo a2c ddpg reinforce --episodes 500 --out runs/compare.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import importlib
from typing import Any, Callable, List, Optional, Sequence, Tuple

import matplotlib

# Headless-friendly default; still works on desktop.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _moving_average(x: Sequence[float], window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) for plotting a moving average with correct x alignment."""
    x_arr = np.asarray(x, dtype=np.float32)
    if window <= 1 or x_arr.size == 0:
        return np.arange(x_arr.size), x_arr
    if x_arr.size < window:
        return np.arange(x_arr.size), x_arr
    kernel = np.ones(window, dtype=np.float32) / float(window)
    y = np.convolve(x_arr, kernel, mode="valid")
    xs = np.arange(window - 1, window - 1 + y.size)
    return xs, y


def _as_two_losses(train_result: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Normalize `Policy.train()` outputs into (loss1, loss2).

    Supported returns:
    - float -> (float, None)
    - tuple/list of length 2 -> (float, float)
    - anything else -> (None, None)
    """
    if train_result is None:
        return None, None
    if isinstance(train_result, (int, float, np.floating)):
        return float(train_result), None
    if isinstance(train_result, (tuple, list)) and len(train_result) == 2:
        a, b = train_result
        return (None if a is None else float(a)), (None if b is None else float(b))
    return None, None


def _mkdir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _make_cartpole_env(env_seed: Optional[int] = None):
    from envs.cartpole_env import CartPoleEnv

    env = CartPoleEnv()

    # If requested, inject a deterministic-but-changing seed each reset() call.
    if env_seed is not None:
        orig_reset = env.reset
        counter = {"i": 0}

        def seeded_reset(seed=None, options=None):
            counter["i"] += 1
            return orig_reset(seed=int(env_seed) + counter["i"], options=options)

        env.reset = seeded_reset  # type: ignore[method-assign]

    return env


_POLICY_SPECS = {
    "a2c": ("policies.a2c", "A2C"),
    "ppo": ("policies.ppo", "PPO"),
    "ddpg": ("policies.ddpg", "DDPG"),
    "reinforce": ("policies.reinforce", "Reinforce"),
}


def _load_policy_cls(policy_name: str):
    if policy_name not in _POLICY_SPECS:
        raise KeyError(f"Unknown policy: {policy_name}")
    module_name, cls_name = _POLICY_SPECS[policy_name]
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, cls_name)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependencies required to run policies. "
            "This project expects packages like `jax`, `flax`, `optax`, `gymnasium`, `mujoco`, "
            "plus `numpy` and `matplotlib`.\n\n"
            "Install the missing package(s) and retry."
        ) from e


@dataclass
class RunResult:
    policy_name: str
    returns: List[float]
    loss1: List[float]
    loss2: List[float]


def run_policy(
    *,
    make_env: Callable[[], Any],
    policy_name: str,
    episodes: int,
    warmup_episodes: int,
) -> RunResult:
    env = make_env()
    policy_cls = _load_policy_cls(policy_name)
    agent = policy_cls(env)

    returns: List[float] = []
    loss1: List[float] = []
    loss2: List[float] = []

    for ep in range(int(episodes)):
        if policy_name == "ddpg" and ep < int(warmup_episodes):
            episode_return = agent.store_episode(random=True)
        else:
            episode_return = agent.store_episode()

        returns.append(float(episode_return))

        # PPO supports train(iterations=...), but defaults are fine for now.
        train_out = agent.train()
        l1, l2 = _as_two_losses(train_out)
        if l1 is not None:
            loss1.append(l1)
        if l2 is not None:
            loss2.append(l2)

    if hasattr(env, "close"):
        env.close()

    return RunResult(policy_name=policy_name, returns=returns, loss1=loss1, loss2=loss2)


def plot_comparison(
    results: Sequence[RunResult],
    *,
    title: str,
    smooth_window: int,
    summary_last: int,
    out_path: str,
) -> None:
    fig, (ax_curve, ax_bar) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})

    # Returns curves
    for r in results:
        y = np.asarray(r.returns, dtype=np.float32)
        x = np.arange(y.size)
        ax_curve.plot(x, y, alpha=0.20, linewidth=1)
        xs, ys = _moving_average(y, smooth_window)
        ax_curve.plot(xs, ys, linewidth=2, label=r.policy_name)

    ax_curve.set_title(title)
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Episode return")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(ncol=2, fontsize=9)

    # Summary bars: mean ± std over last K episodes
    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for r in results:
        k = min(int(summary_last), len(r.returns))
        tail = np.asarray(r.returns[-k:], dtype=np.float32) if k > 0 else np.asarray([], dtype=np.float32)
        labels.append(r.policy_name)
        means.append(float(tail.mean()) if tail.size else float("nan"))
        stds.append(float(tail.std()) if tail.size else 0.0)

    xs = np.arange(len(labels))
    ax_bar.bar(xs, means, yerr=stds, capsize=4, alpha=0.8)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel(f"mean±std (last {summary_last})")
    ax_bar.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _mkdir_for_file(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare policies and save a plot image.")
    p.add_argument("--env", default="cartpole", choices=["cartpole"], help="Environment to run.")
    p.add_argument(
        "--policies",
        nargs="+",
        default=["ppo", "a2c", "ddpg", "reinforce"],
        choices=sorted(_POLICY_SPECS.keys()),
    )
    p.add_argument("--episodes", type=int, default=500, help="Episodes per policy.")
    p.add_argument("--warmup_episodes", type=int, default=10, help="Warmup episodes (used by DDPG random actions).")
    p.add_argument("--smooth_window", type=int, default=25, help="Moving-average window for plotting.")
    p.add_argument("--summary_last", type=int, default=50, help="Episodes to summarize in bar chart.")
    p.add_argument("--env_seed", type=int, default=None, help="If set, seeds reset() with env_seed+episode.")
    p.add_argument("--out", default="runs/policy_comparison.png", help="Output image path (png).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Helpful for WSL/headless setups; doesn't override user choice if already set.
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    args = parse_args(argv)

    if args.env == "cartpole":
        make_env = lambda: _make_cartpole_env(env_seed=args.env_seed)  # noqa: E731
    else:
        raise ValueError(f"Unknown env: {args.env}")

    results: List[RunResult] = []
    for name in args.policies:
        print(f"[compare] Running {name} for {args.episodes} episodes...")
        results.append(
            run_policy(
                make_env=make_env,
                policy_name=name,
                episodes=args.episodes,
                warmup_episodes=args.warmup_episodes,
            )
        )

    title = f"Policy comparison on {args.env} (episodes={args.episodes})"
    plot_comparison(
        results,
        title=title,
        smooth_window=args.smooth_window,
        summary_last=args.summary_last,
        out_path=args.out,
    )

    print(f"[compare] Saved plot to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
