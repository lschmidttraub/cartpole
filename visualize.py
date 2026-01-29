import os
import time
import numpy as np

def visualize_episode(env, policy, *, max_steps=1000, seed=None, deterministic=True, render_fps=60):
    """
    Visualize a single rollout of `policy` in a MuJoCo environment.

    Requirements:
    - `policy.act(obs, train=...)` returns an action (NumPy/JAX array is fine).
    - `env` is a gymnasium-like env with `reset()` and `step()`.
    - For native MuJoCo visualization, `env` should expose `env.model` and `env.data`.

    Returns:
        float: episode return (sum of rewards).
    """
    # Helpful for WSL/headless setups; doesn't override user choice if already set.
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0

    def _policy_action(o):
        a = policy.act(o, train=not deterministic)
        a = np.asarray(a, dtype=np.float32)
        if hasattr(env, "action_space"):
            # Clip to valid range for Box spaces.
            low = getattr(env.action_space, "low", None)
            high = getattr(env.action_space, "high", None)
            if low is not None and high is not None:
                a = np.clip(a, low, high)
        return a

    # Preferred: use MuJoCo's interactive viewer if we have model/data.
    if hasattr(env, "model") and hasattr(env, "data"):
        import mujoco
        import mujoco.viewer

        model = env.model
        data = env.data

        sleep_dt = 0.0 if render_fps is None else (1.0 / float(render_fps))

        launch_passive = getattr(mujoco.viewer, "launch_passive", None)
        if launch_passive is None:
            raise RuntimeError(
                "This MuJoCo build doesn't provide mujoco.viewer.launch_passive(). "
                "Please upgrade mujoco, or use mujoco.viewer.launch(...) manually."
            )

        with launch_passive(model, data) as viewer:
            for _ in range(int(max_steps)):
                is_running = getattr(viewer, "is_running", None)
                if callable(is_running) and not is_running():
                    break

                action = _policy_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)

                viewer.sync()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

                if terminated or truncated:
                    break

        return episode_return

    # Fallback: step the env and call env.render() if available.
    for _ in range(int(max_steps)):
        action = _policy_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        if hasattr(env, "render"):
            env.render()
        if terminated or truncated:
            break

    return episode_return
