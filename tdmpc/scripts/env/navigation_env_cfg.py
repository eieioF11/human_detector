import io
import os
from omegaconf import OmegaConf
import numpy as np

cfg_path = os.path.dirname(__file__).replace(
    "env", "").replace("scripts", "cfg")
usd_path = os.path.dirname(__file__).replace(
    "env", "").replace("scripts", "usd")
config = OmegaConf.load(os.path.join(cfg_path, "env", "env_parameter.yaml"))

class NavigationEnvCfg():
    seed: int = 42
    decimation = 1  # 2
    # limits
    action_limit = [-1.0, 1.0]  # action limit for the robot
    observation_limit = [-1.0, 1.0]  # observation limit for the robot
    # space settings
    action_space = config.robot.action_space
    observation_space = config.robot.observation_space
    robot_observation_space = config.robot.robot_observation_space
    human_observation_space = config.robot.human_observation_space
    state_space = config.robot.state_space
    action_scale = config.robot.action_scale  # scale for action
    # world settings
    num_envs = 1
    dt = config.world.dt  # [s]
    distance_max = config.limits.distance_max  # [m] 90.0
    distance_min = config.limits.distance_min  # [m]
    angular_vel_limit = config.limits.angular_vel_limit  # [rad/s]
    world_mode = config.world.mode
    # collision_limit = 0.3  # [m]
    collision_limit = 0.6  # [m]
    # start
    start_random_generation = config.start.random_generation
    start_random_x_limit = config.start.random_x_range
    start_random_y_limit = config.start.random_y_range
    # goal
    goal_random_generation = config.goal.random_generation
    goal_random_x_limit = config.goal.random_x_range
    goal_random_y_limit = config.goal.random_y_range
    # obstacles
    # static obstacles
    random_obstacle_position = config.obstacles.random_position
    random_obstacle_x_range = config.obstacles.random_x_range
    random_obstacle_y_range = config.obstacles.random_y_range
    # dynamic obstacles
    dynamic_obstacle_dim = 4  # (r, theta, v, w)
    dynamic_obstacles_num = config.robot.dynamic_obstacles_num
    dynamic_obstacle_vel_min = config.limits.dynamic_obstacle_vel_min  # [m/s], [rad/s]
    dynamic_obstacle_vel_max = config.limits.dynamic_obstacle_vel_max  # [m/s], [rad/s]
    near_dynamic_obstacle_dist_limit = 1.0  # [m]
    dynamic_obstacle_goals = 50
    dynamic_obstacle_goal_std = [0.35, 0.7, 0.0]
    scan_step_num = config.robot.scan_step_num
    scan_stack_size = 10
    # reward
    max_reward = config.rewards.max_reward
    min_reward = config.rewards.min_reward

    range_limits = config.rewards.range_limits
    # reward weights
    reward_w_target_distance = config.rewards_weights.target_distance
    reward_w_target_arrival = config.rewards_weights.target_arrival
    reward_w_range_reward = config.rewards_weights.range_reward
    # reward_w_range_reward = config.rewards_weights.range_reward
    reward_w_collision = config.rewards_weights.collision
    reward_w_timeout = config.rewards_weights.timeout
    reward_w_liner_velocity = config.rewards_weights.liner_velocity
    reward_w_angular_velocity = config.rewards_weights.angular_velocity
    reward_w_near_dynamic_obstacle = 0.0
    reward_w_fov = 0.0
    # episode settings
    episode_length_s = 1  # [s]
    max_episode_length = 1  # [steps]
    # other settings
    figure_pixels = config.irsim.figure_pixels
    display = config.display
    logger = None
    irsim_config_path = os.path.join(
        cfg_path, "irsim", config.irsim.config_file)

class IRSimEnvCfg():
    start=[]
    goal=[]
    objects=[] # TODO: add objects

# utility functions
def set_navigation_env_cfg(cfg: NavigationEnvCfg, load_config: OmegaConf):
    cfg.action_space = load_config.robot.action_space
    cfg.observation_space = load_config.robot.observation_space
    cfg.state_space = load_config.robot.state_space
    cfg.action_scale = load_config.robot.action_scale  # scale for action
    cfg.figure_pixels = load_config.irsim.figure_pixels
    cfg.display = load_config.display
    cfg.dynamic_obstacles_num = load_config.robot.dynamic_obstacles_num
    cfg.scan_step_num = load_config.robot.scan_step_num
    cfg.range_limits = load_config.rewards.range_limits
    cfg.reward_w_target_distance = load_config.rewards_weights.target_distance
    cfg.reward_w_target_arrival = load_config.rewards_weights.target_arrival
    cfg.reward_w_range_reward = load_config.rewards_weights.range_reward
    cfg.reward_w_collision = load_config.rewards_weights.collision
    cfg.reward_w_timeout = load_config.rewards_weights.timeout
    cfg.reward_w_liner_velocity = load_config.rewards_weights.liner_velocity
    cfg.reward_w_angular_velocity = load_config.rewards_weights.angular_velocity
    cfg.irsim_config_path = os.path.join(
        cfg_path, "irsim", load_config.irsim.config_file)
    irsim_cfg = OmegaConf.load(cfg.irsim_config_path)
    # set world dt
    print("irsim dt:", irsim_cfg.world.step_time)
    load_config.world.dt = irsim_cfg.world.step_time
    cfg.dt = load_config.world.dt  # [s]
    cfg.num_envs = load_config.world.num
    # start
    cfg.start_random_generation = load_config.start.random_generation
    cfg.start_random_x_limit = load_config.start.random_x_range
    cfg.start_random_y_limit = load_config.start.random_y_range
    # goal
    cfg.goal_random_generation = load_config.goal.random_generation
    cfg.goal_random_x_limit = load_config.goal.random_x_range
    cfg.goal_random_y_limit = load_config.goal.random_y_range
    # obstacles
    cfg.random_obstacle_position = load_config.obstacles.random_position
    cfg.random_obstacle_x_range = load_config.obstacles.random_x_range
    cfg.random_obstacle_y_range = load_config.obstacles.random_y_range

    # limits
    cfg.distance_max = load_config.limits.distance_max  # [m] 90.0
    cfg.distance_min = load_config.limits.distance_min  # [m]
    cfg.dynamic_obstacle_vel_min = load_config.limits.dynamic_obstacle_vel_min  # [m/s], [rad/s]
    cfg.dynamic_obstacle_vel_max = load_config.limits.dynamic_obstacle_vel_max
    try:
        cfg.angular_vel_limit = load_config.limits.angular_vel_limit
    except AttributeError:
        cfg.angular_vel_limit = cfg.action_scale[2]
    cfg.robot_observation_space = load_config.robot.robot_observation_space
    cfg.human_observation_space = load_config.robot.human_observation_space
    try:
        cfg.world_mode = load_config.world.mode
    except AttributeError:
        cfg.world_mode = "normal"
    try:
        cfg.reward_w_near_dynamic_obstacle = load_config.rewards_weights.near_dynamic_obstacle
    except AttributeError:
        cfg.reward_w_near_dynamic_obstacle = 0.0
    try:
        cfg.scan_stack_size = load_config.robot.scan_stack_size
    except AttributeError:
        pass
    try:
        cfg.reward_w_fov = load_config.rewards_weights.fov
    except AttributeError:
        cfg.reward_w_fov = 0.0
    try:
        cfg.near_dynamic_obstacle_dist_limit = load_config.limits.near_dynamic_obstacle_dist_limit
    except AttributeError:
        cfg.near_dynamic_obstacle_dist_limit = 1.0



