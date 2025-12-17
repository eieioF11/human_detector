import gymnasium as gym
import numpy as np

from envs.wrappers.timeout import Timeout

GYMNASIUM_TASKS = {
    "gymnasium-bipedalwalker": "BipedalWalker-v3",
    "gymnasium-pendulum": "Pendulum-v1",
}


class GymnasiumWrapper(gym.Wrapper):
    """
    Wrapper for Gymnasium environments.
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._cumulative_reward: float = 0.0

    def reset(self, **kwargs):
        self._cumulative_reward = 0.0
        # Gymnasiumのresetは (obs, info) を返す
        obs, _ = self.env.reset(seed=self.cfg.seed, **kwargs)
        return obs

    def step(self, action: np.ndarray):
        # Gymnasiumのstepは (obs, reward, terminated, truncated, info) を返す
        obs, reward, terminated, truncated, info = self.env.step(action.copy())
        self._cumulative_reward += float(reward)
        done = terminated or truncated
        info["terminated"] = terminated
        info["truncated"] = truncated
        # BipedalWalkerの成功条件の例 (必要に応じて調整)
        if self.cfg.task == "gymnasium-bipedalwalker" and "success" not in info:
            # BipedalWalkerには明確な成功報酬がないため、
            # ここでは単純にエピソード完了を成功とみなすか、
            # 特定の累積報酬閾値を設けるなどの対応が
            # 必要です。
            # 今回は、Timeoutラッパーでエピソード長が管理されるため、
            # ここでは特に何もしません。
            pass
        return obs, float(reward), done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


def make_env(cfg):
    """
    Make Gymnasium environment.
    """
    if cfg.task not in GYMNASIUM_TASKS:
        raise ValueError(f"Unknown Gymnasium task: {cfg.task}")
    assert cfg.obs == "state", "This task only supports state observations."

    env_id = GYMNASIUM_TASKS[cfg.task]
    # render_modeはcfgから取得するか、デフォルト値を設定
    render_mode = getattr(cfg, "render_mode", "rgb_array")
    env = gym.make(env_id, render_mode=render_mode)
    env.action_space.seed(cfg.seed)

    env = GymnasiumWrapper(env, cfg)
    env = Timeout(
        env,
        max_episode_steps=getattr(cfg, "max_episode_steps", 1600),
    )

    return env
