import warnings

import gymnasium as gym

from envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(
        f"Missing dependencies for task {task}; install dependencies to use this environment."
    )


try:
    from envs.gymnasium_envs import make_env as make_gymnasium_env
except ImportError:  # Changed from bare except
    make_gymnasium_env = missing_dependencies


warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)

    # Directly use make_gymnasium_env as other environments are not needed
    try:
        env = make_gymnasium_env(cfg)
    except ValueError as e:  # Catch specific error
        raise ValueError(
            f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists. Original error: {e}'
        )

    if (
        env is None
    ):  # Should not happen if make_gymnasium_env raises ValueError on failure
        raise ValueError(
            f'Failed to make environment "{cfg.task}": make_gymnasium_env returned None.'
        )

    env = TensorWrapper(env)
    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except AttributeError:  # Box - Changed from bare except
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
