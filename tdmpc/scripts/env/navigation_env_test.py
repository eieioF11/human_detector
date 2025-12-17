import io
from omegaconf import OmegaConf
import numpy as np
from env.custom_behavior_methods import *
import irsim
import cv2
import gymnasium as gym
from gymnasium.spaces import Box
from collections import defaultdict
from termcolor import colored
from numba import njit
from env.navigation_env_cfg import NavigationEnvCfg, IRSimEnvCfg
from env.math_utils import (
    PI,
    set_all_seeds,
    normalize_value,
    normalize_polar,
    calc_local_polar,
    random_state,
    np_norm
)
from env.utility import *


class NavigationEnv(gym.Env):
    cfg: NavigationEnvCfg
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 15}

    def __init__(self, cfg: NavigationEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the Navigation Environment.
        Args:
            cfg (NavigationEnvCfg): Configuration for the environment.
            render_mode (str | None): The rendering mode for the environment.
        Keyword Args:
            debug (bool): If True, enable debug mode in the environment and the robot will move automatically.
        Raises:
            AssertionError: If logger is not set in NavigationEnvCfg.
        """
        super().__init__()
        self.render_mode = render_mode
        self.debug = kwargs.get("debug", False)
        self._render = kwargs.get("render", cfg.display)
        title = kwargs.get("title", None)
        self.logger = cfg.logger
        self.cfg = cfg
        assert self.logger is not None, "Logger is not set in NavigationEnvCfg"
        self.logger.info("- NavigationEnv initializing - -")
        self.logger.info(f"render_mode: {self.render_mode}")
        self.logger.info(f"debug: {self.debug}")
        self.logger.info(f"irsim display: {self.cfg.display}")
        self.random_goal = self.cfg.goal_random_generation
        irsim_cfg = OmegaConf.load(cfg.irsim_config_path)
        irsim_cfg.world.figure_pixels = self.cfg.figure_pixels
        self.figure_pixels = irsim_cfg.world.figure_pixels
        self.logger.info(f"figure_pixels: {irsim_cfg.world.figure_pixels}")
        # set action and observation space
        self.metadata["render_fps"] = int(np.round(1.0/self.cfg.dt))
        self.logger.info(f"render_fps: {self.metadata['render_fps']}")
        # make irsim environment
        self._shortest_length = np.zeros((self.cfg.num_envs), dtype=np.float32)
        self._gen_multiple_worlds(num_envs=self.cfg.num_envs, title=title)
        self._state = self.env.get_robot_state().T[0]
        self._pre_state = self._state
        self._mileage = 0.0
        self.buf = io.BytesIO()
        # gym IO
        self.action_space = Box(
            low=self.cfg.action_limit[0], high=self.cfg.action_limit[1], shape=(self.cfg.action_space,), dtype=np.float32)
        self.logger.info(f"action_space: {self.action_space}")
        self.observation_space = Box(
            low=self.cfg.observation_limit[0], high=self.cfg.observation_limit[1], shape=(self.cfg.observation_space,), dtype=np.float32)
        self.logger.info(f"observation_space: {self.observation_space}")

        self._scan_num = irsim_cfg.robot[0]['sensors'][0]["number"]
        self._scan_range_min = irsim_cfg.robot[0]['sensors'][0]["range_min"]
        self._scan_range_max = irsim_cfg.robot[0]['sensors'][0]["range_max"]
        self._distance_min = self.cfg.distance_min
        self._distance_max = self.cfg.distance_max
        self._dynamic_obstacle_min = self.cfg.distance_min
        self._dynamic_obstacle_max = self.cfg.distance_max
        self._dynamic_obstacle_vel_min = self.cfg.dynamic_obstacle_vel_min
        self._dynamic_obstacle_vel_max = self.cfg.dynamic_obstacle_vel_max

        self._state = self.env.get_robot_state().T[0]
        self._target_polar = calc_local_polar(
            self.env.robot.goal.T[0], self._state)
        self._pre_target_polar = self._target_polar
        self._dynamic_obstacles = np.zeros(
            (self.cfg.dynamic_obstacles_num, self.cfg.dynamic_obstacle_dim), dtype=np.float32)
        self._episode_length_buf = np.zeros((1), dtype=np.long)
        self.logger.info(f"irsim_config scan_num: {self._scan_num}")
        self._action = np.zeros((self.cfg.action_space), dtype=np.float32)
        self._scan = np.zeros((self._scan_num), dtype=np.float32)
        # indices for scan reduction
        self._indices = np.linspace(
            0, self._scan_num-1, num=self.cfg.scan_step_num, dtype=np.int64)
        self._scan_tmp = None
        self._decoded_obs = None
        self._collision_off = False
        self._get_env_state()  # get the current state of the environment
        self.logger.info("- NavigationEnv initialized - -")

    def irsim_env(self):
        return self.env

    def base_controller(self, normalization = False) -> np.ndarray:
        base_action = self.env.robot.obj_behavior.gen_vel(
                self.env.robot.ego_object,  self.env.robot.external_objects
            )
        if normalization:
            base_action = base_action / self.cfg.action_scale
        return base_action

# obs, reward, done, truncated, info
    def step(self, action):
        """
        Step the environment with the given action.
        Args:
            action (np.ndarray): The action to take. Dimension is 3 (vx, vy, omega).
        Returns:
            observation (np.ndarray): The observation of the environment.
            reward (float): The reward received from the environment.
            done (bool): Whether the episode is done.
            truncated (bool): Whether the episode is truncated due to timeout.
            info (dict): Additional information about the environment.
        """
        robot_id = 0
        self.logger.debug(f"env_id: {self._env_id}")
        # scaleing action
        action = action * self.cfg.action_scale  # scale the action
        self._action = action
        self.logger.debug(f"action: {self._action}")
        env_action = action.T
        self.logger.debug(f"debug mode: {self.debug}")
        # env step
        if self.debug:  # debug
            self.env.step()  # auto move robot
        else:
            self.env.step(action_id=robot_id, action=env_action)
        if self._decoded_obs is not None:
            self.plot_decoded_obs(self._decoded_obs)
        # visualization
        if self.env.display and self._render:
            self.env.render(interval=0.001)
            self.env._env_plot.fig.savefig(self.buf, format='png')
        # cal length of episode
        self._episode_length_buf += 1
        truncated = self._episode_length_buf >= self.cfg.max_episode_length
        # get the current state of the environment
        self._get_env_state()
        self.logger.debug(
            f"target_pos map:{self.env.robot.goal.T[0]} polar:{self._target_polar}")
        # self._mileage += np.linalg.norm(self._state - self._pre_state)
        self._mileage += np_norm(self._state, self._pre_state)
        # calc observation and reward
        observation = self._observation()
        reward = self._reward(truncated)
        info = defaultdict(float)
        info["success"] = self.env.robot.arrive
        info["truncated"] = truncated
        info["collision"] = self._collision
        info["env_id"] = self._env_id
        info["shortest_length"] = self._shortest_length[self._env_id]
        info["mileage"] = self._mileage
        # calc done
        done = self.done()
        if (done):
            self._reset_done()
        # debug logging
        self.logger.debug(f"observation:\n{observation}")
        self.logger.debug(
            f"episode_length_buf: {self._episode_length_buf}/{self.cfg.max_episode_length} arrive: {self.env.robot.arrive}")
        self.logger.debug(
            f"reward: {reward}, done: {done}, truncated: {truncated}")
        self.logger.debug("mileage: {:.2f}".format(self._mileage))
        return observation, reward, done, truncated, info

    def done (self) -> bool:
        if self._collision_off:
            return self.env.robot.arrive
        return self.env.done()

    def collision(self) -> bool:
        if self._collision_off:
            return False
        else:
            self.logger.debug("Collision detected (irsim)")
            return self.env.robot.collision

    def _reward(self, truncated) -> np.ndarray:
        # print(f"ped_vel: {ped_vel}, ped_state: {ped_state}")
        # print(f"self.cfg.distance_max: {self.cfg.distance_max} {self._dynamic_obstacles[0, 0]*self.cfg.distance_max} {self.cfg.near_dynamic_obstacle_dist_limit}")
        return self.compute_rewards(
            w_target_distance=self.cfg.reward_w_target_distance,
            w_target_arrival=self.cfg.reward_w_target_arrival,
            w_range_reward=self.cfg.reward_w_range_reward,
            w_collision=self.cfg.reward_w_collision,
            w_timeout=self.cfg.reward_w_timeout,
            w_liner_velocity=self.cfg.reward_w_liner_velocity,
            w_angular_velocity=self.cfg.reward_w_angular_velocity,
            w_near_dynamic_obstacle=self.cfg.reward_w_near_dynamic_obstacle,
            w_fov=self.cfg.reward_w_fov,
            max_reward=self.cfg.max_reward,
            min_reward=self.cfg.min_reward,
            range_limits=self.cfg.range_limits,
            angular_vel_limit=self.cfg.angular_vel_limit,  # angular velocity limit
            action=self._action,  # action
            target_dist=self._target_polar[0],  # target distance
            # previous target distance
            pre_target_dist=self._pre_target_polar[0],
            scan=self._scan,  # lidar scan
            arrive=self.env.robot.arrive,
            collision=self._collision,  # collision
            truncated=truncated,  # timeout
            distance_max=self.cfg.distance_max,
            near_dynamic_obstacle_dist=self._dynamic_obstacles[0, 0],
            near_dynamic_obstacle_dist_limit=self.cfg.near_dynamic_obstacle_dist_limit,
            ped_fov=self._dynamic_obstacles_fov[0]
        )

    @staticmethod
    @njit(cache=True)
    def compute_rewards(
        w_target_distance: float,
        w_target_arrival: float,
        w_range_reward: float,
        w_collision: float,
        w_timeout: float,
        w_liner_velocity: float,
        w_angular_velocity: float,
        w_near_dynamic_obstacle: float,
        w_fov: float,
        max_reward: float,
        min_reward: float,
        range_limits: float,
        angular_vel_limit: float,
        action: np.ndarray,
        target_dist: float,
        pre_target_dist: float,
        scan: np.ndarray,
        distance_max: float,
        near_dynamic_obstacle_dist: float,
        near_dynamic_obstacle_dist_limit: float,
        ped_fov: bool,
        collision: bool,
        arrive: bool,
        truncated: bool
    ) -> np.ndarray:
        arrival_reward = w_target_arrival * arrive
        distance_reward = (pre_target_dist - target_dist) * w_target_distance

        collision_reward = w_collision * collision
        min_range = np.min(scan)
        def r3(x): return range_limits - x if x < range_limits else 0.0
        range_reward = w_range_reward * r3(min_range)

        # action rewards
        liner = w_liner_velocity * action[0]
        angular = 0.0
        if np.abs(action[1]) > angular_vel_limit:
            angular = w_angular_velocity * np.abs(action[1])

        time_out_reward = w_timeout * truncated

        ped_reward = 0.0
        near_dynamic_obstacle_reward = ped_fov * w_fov
        if ((near_dynamic_obstacle_dist*distance_max) < near_dynamic_obstacle_dist_limit):
            near_dynamic_obstacle_reward = w_near_dynamic_obstacle * \
                np.exp(-near_dynamic_obstacle_dist)
        # near_dynamic_obstacle_reward = w_near_dynamic_obstacle * (1.0 - near_dynamic_obstacle_dist) + ped_fov * w_fov
        reward = distance_reward + collision_reward + range_reward + \
            liner + angular + arrival_reward + time_out_reward + \
            ped_reward + near_dynamic_obstacle_reward
        return np.clip(reward, min_reward, max_reward)

    def _get_env_state(self):
        scan = self.env.get_lidar_scan()  # get the lidar scan dat
        state = self.env.get_robot_state()  # get the current state of the environment
        self.logger.debug(f"{self.cfg.dynamic_obstacles_num}")
        self._dynamic_obstacles, self._dynamic_obstacles_fov, self._dynamic_obstacles_obs = get_dynamic_obstacles_custom(
            self.env,
            self.env.obstacle_list,
            self.cfg.dynamic_obstacles_num,
            self.cfg.dynamic_obstacle_dim,
            state=self._state,
            dynamic_obstacle_min=self._dynamic_obstacle_min,
            dynamic_obstacle_max=self._dynamic_obstacle_max,
            dynamic_obstacle_vel_min=self._dynamic_obstacle_vel_min,
            dynamic_obstacle_vel_max=self._dynamic_obstacle_vel_max
        )
        self.logger.debug(
            f"dynamic_obstacles:\n{self._dynamic_obstacles} \n fov: {self._dynamic_obstacles_fov} \n obs: {self._dynamic_obstacles_obs}")
        self._pre_state = self._state
        self._state = state.T[0]
        self._scan = scan["ranges"]
        # self._collision = self.env.robot.collision  # get the collision status of the robot
        self._collision = self.collision()  # get the collision status of the robot
        if self._scan_tmp is None:
            self._scan_tmp = np.zeros(
                (self.cfg.scan_stack_size, len(self._scan)), dtype=np.float32)
        else:
            self._scan_tmp[0] = self._scan
            self._scan_tmp = np.roll(self._scan_tmp, -1, axis=0)
        # set target position
        self._pre_target_polar = self._target_polar
        self._target_polar = calc_local_polar(
            self.env.robot.goal.T[0], self._state)

    @staticmethod
    @njit(cache=True)
    def make_observation(
        # target_pos: np.ndarray,
        target_polar: np.ndarray,
        scan: np.ndarray,
        scan_tmp: np.ndarray,
        dynamic_obstacles: np.ndarray,
        indices: np.ndarray,
        scan_range_min: float,
        scan_range_max: float,
        distance_min: float,
        distance_max: float,
    ) -> np.ndarray:
        # reduce the scan to the configured number of steps
        reduced_scan = normalize_value(
            scan[indices], min_value=scan_range_min, max_value=scan_range_max)
        # scan_tmp = normalize_value(
        #     scan_tmp, min_value=scan_range_min, max_value=scan_range_max)
        # reduced_scan = scan_tmp[:, indices].flatten()
        # normalize the polar coordinates
        target_polar = normalize_polar(
            target_polar, distance_min=distance_min, distance_max=distance_max)
        # flatten the dynamic obstacles
        dynamic_obstacles_vec = dynamic_obstacles.flatten()
        return np.concatenate((
            target_polar.astype(np.float32),
            reduced_scan.astype(np.float32),
            dynamic_obstacles_vec.astype(np.float32)
        ))

    def _observation(self) -> np.ndarray:
        return self.make_observation(
            target_polar=self._target_polar,
            scan=self._scan,
            scan_tmp=self._scan_tmp,
            dynamic_obstacles=self._dynamic_obstacles,
            # dynamic_obstacles=self._dynamic_obstacles_obs,
            indices=self._indices,
            scan_range_min=self._scan_range_min,
            scan_range_max=self._scan_range_max,
            distance_min=self._distance_min,
            distance_max=self._distance_max
        )

    # random generation
    def _random_start_generation(self, env, x_limit, y_limit):
        self.logger.info("Generating random start position")
        env_action = np.array([[0.0], [0.0],])
        while True:
            rand_start = random_state(x_limit, y_limit)
            env.robot.set_state(
                state=rand_start,
                init=True,
            )
            self.logger.debug(f"start_pos: {rand_start.T[0]}")
            env.reset()
            env.step(action_id=0, action=env_action)
            if not env.robot.collision:
                break

    # generate worlds
    def _gen_multiple_worlds(self, num_envs: int, title=None):
        self._env_id = 0
        self._envs = []
        self._env_cfgs = []
        env = irsim.make(
            self.cfg.irsim_config_path,
            display=self.cfg.display,
            save_ani=False,
            # default is "INFO", set to "ERROR" to reduce log output (INFO,WARNING,ERROR)
            log_level="ERROR",
        )
        if title is not None:
            env.set_title(title)
        env.load_behavior("env.custom_behavior_methods")
        env.reset()
        env.reset_plot()
        if self.cfg.random_obstacle_position:
            self.logger.info("Generating random obstacle position")
            env.random_obstacle_position(
                range_low=[self.cfg.random_obstacle_x_range[0],
                           self.cfg.random_obstacle_y_range[0], -PI],
                range_high=[self.cfg.random_obstacle_x_range[1],
                            self.cfg.random_obstacle_y_range[1], PI],
            )
        self.env = env
        set_all_seeds(self.cfg.seed)
        self._obs_random_init = True
        if self.cfg.world_mode == "cross":
            self.logger.info("World mode: cross")
            cross = True
            self._obs_random_init = True
        else:
            self.logger.info("World mode: normal")
            cross = False
        flip = False
        for i in range(num_envs):
            self.logger.info(f"Generating world {i+1}/{num_envs}")
            env_cfg = IRSimEnvCfg()
            # set random start position
            if self.cfg.start_random_generation:
                if not flip:
                    self._random_start_generation(
                        env=env, x_limit=self.cfg.start_random_x_limit, y_limit=self.cfg.start_random_y_limit)
                else:
                    self._random_start_generation(
                        env=env, x_limit=self.cfg.goal_random_x_limit, y_limit=self.cfg.goal_random_y_limit)
                self.logger.info(
                    f"Generated start_pos: {env.robot.state.T[0]}")
            # set random goal
            if self.cfg.goal_random_generation:
                self.logger.info("Generating random goal position")
                if not flip:
                    env.robot.set_random_goal(
                        obstacle_list=env.obstacle_list,
                        init=True,
                        range_limits=[[self.cfg.goal_random_x_limit[0], self.cfg.goal_random_y_limit[0], -PI], [
                            self.cfg.goal_random_x_limit[1], self.cfg.goal_random_y_limit[1], PI]],
                    )
                else:
                    env.robot.set_random_goal(
                        obstacle_list=env.obstacle_list,
                        init=True,
                        range_limits=[[self.cfg.start_random_x_limit[0], self.cfg.start_random_y_limit[0], -PI], [
                            self.cfg.start_random_x_limit[1], self.cfg.start_random_y_limit[1], PI]],
                    )
                self.logger.info(
                    f"Generated target_pos: {env.robot.goal.T[0]}")
            if cross:
                self._obs_init_state = []
                id = 0
                for obs in self.env.obstacle_list:
                    obs_info = obs.get_obstacle_info()
                    start = np.array(
                        [obs_info.center.T[0, 0], obs_info.center.T[0, 1], 0.0])
                    goal = np.array(
                        [30.0-start[0], start[1], 0.0])
                    if obs.get_info().kinematics is not None:
                        fl = True
                        noize_std = self.cfg.dynamic_obstacle_goal_std
                        noize = np.random.randn(3)*noize_std
                        obs.set_goal(goal=np.array(
                            [goal + noize], dtype=np.float32).T, init=True)
                        print(
                            f"start {start} goal {goal + noize} {obs._init_state.T}")
                        for _ in range(self.cfg.dynamic_obstacle_goals):
                            noize = np.random.randn(3)*noize_std
                            if not fl:
                                v = goal + noize
                                fl = True
                            else:
                                v = start + noize
                                fl = False
                            obs.append_goal(v)
                        noize = np.array([np.random.randn(3)*noize_std]).T
                        self._obs_init_state.append((id, obs._init_state))
                        obs.set_state(state=obs._init_state + noize, init=True)
                    id += 1
                cross = False
                if not flip:
                    flip = True
                else:
                    flip = False
            else:
                flip = False
            env_cfg.start = env.robot.state
            env_cfg.goal = env.robot.goal
            self.logger.info(
                f"start: {env.robot.state.T[0]}, goal: {env.robot.goal.T[0]}")
            self._env_cfgs.append(env_cfg)
            self._shortest_length[i] = np.linalg.norm(
                env.robot.state - env.robot.goal)
        self._env_cfgs = np.array(self._env_cfgs)
        self.logger.info(f"shortest_length: {self._shortest_length}")

    # reset environment

    def _reset_done(self):
        self.logger.info(
            f"arrive: {self.env.robot.arrive}")
        # reset np arrays
        self._action = np.zeros((self.cfg.action_space), dtype=np.float32)
        self._scan = np.zeros((self._scan_num), dtype=np.float32)
        self._mileage = 0.0
        self._state = self.env.get_robot_state().T[0]
        self._pre_state = self._state
        self._target_polar = calc_local_polar(
            self.env.robot.goal.T[0], self._state)
        self._pre_target_polar = self._target_polar
        self._dynamic_obstacles = np.zeros(
            (self.cfg.dynamic_obstacles_num, self.cfg.dynamic_obstacle_dim), dtype=np.float32)
        if self.cfg.world_mode == "cross" and self._obs_random_init:
            for id, init_state in self._obs_init_state:
                obs = self.env.obstacle_list[id]
                noize_std = self.cfg.dynamic_obstacle_goal_std
                noize = np.array([np.random.randn(3)*noize_std]).T
                obs.set_state(state=init_state + noize, init=True)
        # reset environment
        self.env.reset()
        self.env.reset_plot()

    def reset(self, **kwargs):
        self.logger.info("Resetting environment")
        self._render = kwargs.get("render", self._render)
        env_id = kwargs.get("env_id", None)
        state = kwargs.get("state", None)
        self._collision_off = kwargs.get("collision_off", False)
        debug = kwargs.get("debug", None)
        if debug is not None:
            self.debug = debug
        super().reset()
        self._env_id += 1
        if self._env_id >= self.cfg.num_envs:
            self._env_id = 0
        if env_id is not None:
            assert env_id < self.cfg.num_envs, f"env_id {env_id} is out of range. num_envs: {self.cfg.num_envs}"
            self._env_id = env_id
        env_cfg = self._env_cfgs[self._env_id]
        if state is not None:
            self.logger.info(f"Setting state: {state}")
            self.env.robot.set_state(
                state=state,
                init=True,
            )
        else:
            self.env.robot.set_state(
                state=env_cfg.start,
                init=True,
            )
        self.env.robot.set_goal(
            goal=env_cfg.goal,
            init=True,
        )
        self.logger.info(f"env_id: {self._env_id}")
        self._reset_done()
        observation_space = np.zeros(
            (self.cfg.observation_space), dtype=np.float32)
        self.buf = io.BytesIO()
        self._episode_length_buf = np.zeros((1), dtype=np.long)
        if self.env.display and self._render:  # Adding the first frame
            self.env.render(interval=0.001)
            self.env._env_plot.fig.savefig(self.buf, format='png')
        return observation_space, defaultdict(float)

    # render environment
    def render(self, **kwargs):
        """
            Render the environment.
            Returns:
                - RGB image of the environment.
            Raises:
                - NotImplementedError: If the render mode is not supported.
                - ValueError: If the buffer is empty.
            memo:
                - Only supports rgb_array
                - Changing the window size causes an error during video conversion.
                - Resizing solves the problem, but the ratio becomes distorted.
        """
        mode = kwargs.get("mode", self.render_mode)
        if mode != 'rgb_array':
            self.logger.error(colored(
                f"Render mode {mode} is not supported. Use 'rgb_array' for rendering.", "red", attrs=["bold"]))
            raise NotImplementedError()
        if not self.env.display:
            self.logger.error(colored(
                "Environment display is not enabled. Cannot render.", "red", attrs=["bold"]))
            raise NotImplementedError("Environment display is not enabled.")
        if not self._render:
            self.logger.error(colored(
                "Rendering is disabled. Set 'render' to True to enable rendering.", "red", attrs=["bold"]))
            raise NotImplementedError("Rendering is disabled.")
        # self.env.env_plot
        self.logger.debug(f"buffer size: {self.buf.getbuffer().nbytes}")
        if self.buf.getbuffer().nbytes == 0:
            self.logger.error(
                colored("Buffer is empty. Rendering will not work.", "red", attrs=["bold"]))
            raise ValueError("Buffer is empty.")
        enc = np.frombuffer(self.buf.getvalue(),
                            dtype=np.uint8)  # bufferからの読み出し
        dst = cv2.imdecode(enc, 1)  # デコード
        dst = cv2.resize(dst, self.figure_pixels)
        dst = dst[:, :, ::-1]  # BGR->RGB
        self.buf = io.BytesIO()
        return dst
