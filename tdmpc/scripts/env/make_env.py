import gymnasium as gym
from envs.gymnasium_envs import GymnasiumWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.timeout import Timeout
from envs.wrappers.dt_separation import DtSeparation

def make_nav_env(cfg, env_cfg, debug=False, **kwargs):
    """
    Make Gymnasium environment.
    """
    env_id = kwargs.get("env_id", "NavigationEnv-v0")
    use_dt_sep = kwargs.get("use_dt_sep", False)
    print(f"[INFO] env_id: {env_id}")
    env = gym.make(env_id, cfg=env_cfg, render_mode='rgb_array', debug=debug, **kwargs)
    # env = gym.make("NavigationEnv-v0", cfg=env_cfg, render_mode='rgb_array', debug=debug, **kwargs)
    env.action_space.seed(cfg.seed)
    env = GymnasiumWrapper(env, cfg)
    env = TensorWrapper(env)  # Convert numpy arrays to torch tensors
    if use_dt_sep:
        env = DtSeparation(env, cfg.dt, cfg.world["dt"])
    # set variables of config
    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except AttributeError:  # Box - Changed from bare except
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    cfg.r_obs_shape = {"state": (env_cfg.robot_observation_space,)}
    cfg.h_obs_shape = {"state": (env_cfg.human_observation_space,)}
    return env