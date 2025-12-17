import gymnasium as gym


class DtSeparation(gym.Wrapper):
    """
    Wrapper for enforcing a time limit on the environment.
    """

    def __init__(self, env, dt, sim_dt):
        super().__init__(env)
        self._dt = dt
        self._sim_dt = sim_dt
        self._step_per_dt = int(self._dt // self._sim_dt)
        if self._step_per_dt < 1:
            self._step_per_dt = 1
        print(f"dt: {self._dt}, sim_dt: {self._sim_dt}, step_per_dt: {self._step_per_dt}")
        self._render_frames = []
        self._render = False

    @property
    def dt(self):
        return self._dt

    def reset(self, **kwargs):
        self._render = kwargs.get("render", self._render)
        self._render_frames = []
        return self.env.reset(**kwargs)

    def step(self, action):
        sum_reward = 0.0
        for _ in range(self._step_per_dt):
            obs, reward, done, info = self.env.step(action)
            if self._render:
                frame = self.env.render()
                self._render_frames.append(frame)
            sum_reward += reward
            if done:
                break
        return obs, sum_reward, done, info

    def render(self, **kwargs):
        render_frames = self._render_frames
        self._render_frames = []
        return render_frames
        # return self.env.render(**kwargs)
