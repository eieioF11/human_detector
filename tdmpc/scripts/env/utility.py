import math
import numpy as np
import random
from numba import njit
import torch

from env.math_utils import *

@njit(cache=True)
def calc_collision(
    scan: np.ndarray, dynamic_obstacle: np.ndarray, distance_max: float, collision_limit: float
) -> bool:
    min_range = np.min(scan)
    min_ped_dist = dynamic_obstacle * distance_max
    min_dist = min(min_range, min_ped_dist)
    return min_dist < collision_limit

@njit(cache=True)
def get_dynamic_obstacle(
    obstacle_list_len: int,
    dynamic_obstacles_num: int,
    dynamic_obstacle_dim: int,
    state: np.ndarray,
    obstacles_dynamic: np.ndarray,  # [num]
    obstacles_pos: np.ndarray,  # [x, y, theta]
    obstacles_vel: np.ndarray,  # [vx, omega]
    obstacles_fov: np.ndarray,  # [num]
    dynamic_obstacle_min: float,
    dynamic_obstacle_max: float,
    dynamic_obstacle_vel_min: tuple,
    dynamic_obstacle_vel_max: tuple
) -> np.ndarray:
    VALUE = 255.0
    num = obstacle_list_len if obstacle_list_len > dynamic_obstacles_num else dynamic_obstacles_num
    dynamic_obstacles = np.zeros(
        (num, dynamic_obstacle_dim), dtype=np.float32)
    dynamic_obstacles_obs = np.zeros(
        (num, 2), dtype=np.float32)
    dynamic_obstacles[:, 0] = VALUE
    count = 0
    for i in range(obstacle_list_len):
        if obstacles_dynamic[i]:  # if the obstacle is dynamic
            dynamic_obs_pos_polar = calc_local_polar(
                obstacles_pos[i], state)  # [r, theta] local
            # normalize
            dynamic_obs_pos_polar = normalize_polar(
                dynamic_obs_pos_polar, dynamic_obstacle_min, dynamic_obstacle_max)
            obstacles_vel[i, 0] = normalize_value(
                np.array([obstacles_vel[i, 0]]), min_value=dynamic_obstacle_vel_min[0], max_value=dynamic_obstacle_vel_max[0])[0]
            obstacles_vel[i, 1] = normalize_value(
                np.array([obstacles_vel[i, 1]]), min_value=dynamic_obstacle_vel_min[1], max_value=dynamic_obstacle_vel_max[1], out_min=-1.0, out_max=1.0)[0]
            dynamic_obstacles[count] = np.concatenate(
                (dynamic_obs_pos_polar, obstacles_vel[i]))
            dynamic_obstacles_obs[count] = dynamic_obs_pos_polar
            count += 1
    sort_index = np.argsort(dynamic_obstacles[:, 0])
    dynamic_obstacles = dynamic_obstacles[sort_index]  # sort by distance
    obstacles_fov = obstacles_fov[sort_index]
    condition = (dynamic_obstacles[:, 0] == VALUE)
    dynamic_obstacles[condition, 0] = 0  # replace VALUE with 0
    # select only the first dynamic_obstacles_num obstacles
    dynamic_obstacles = dynamic_obstacles[0:dynamic_obstacles_num]
    return dynamic_obstacles, obstacles_fov[0:dynamic_obstacles_num], dynamic_obstacles_obs[0:dynamic_obstacles_num]

def get_dynamic_obstacles(
    env,
    obstacle_list: list,
    dynamic_obstacles_num: int,
    dynamic_obstacle_dim: int,
    state: np.ndarray,
    dynamic_obstacle_min: float,
    dynamic_obstacle_max: float,
    dynamic_obstacle_vel_min: tuple,
    dynamic_obstacle_vel_max: tuple
) -> np.ndarray:
    obstacle_list_len = len(obstacle_list)
    obstacles_dynamic = np.zeros((obstacle_list_len), dtype=np.bool_)
    obstacles_pos = np.zeros((obstacle_list_len, 2), dtype=np.float32)
    obstacles_vel = np.zeros((obstacle_list_len, 2), dtype=np.float32)
    obstacles_fov = np.zeros((obstacle_list_len), dtype=np.bool_)
    for i in range(obstacle_list_len):
        obs = obstacle_list[i]
        obs_info = obs.get_obstacle_info()
        obstacles_dynamic[i] = obs.get_info().kinematics is not None
        obstacles_pos[i] = obs_info.center.T[0]
        obstacles_vel[i] = obs_info.velocity.T[0]
        obstacles_fov[i] = False
        if obs.get_info().kinematics is not None:
            # print(f"{obs.get_fov_detected_objects()}")
            obstacles_fov[i] = obs.fov_detect_object(env.robot)
    dynamic_obstacles, obstacles_fov, dynamic_obstacles_obs = get_dynamic_obstacle(
        obstacle_list_len=obstacle_list_len,
        dynamic_obstacles_num=dynamic_obstacles_num,
        dynamic_obstacle_dim=dynamic_obstacle_dim,
        state=state,
        obstacles_dynamic=obstacles_dynamic,
        obstacles_pos=obstacles_pos,
        obstacles_vel=obstacles_vel,
        obstacles_fov=obstacles_fov,
        dynamic_obstacle_min=dynamic_obstacle_min,
        dynamic_obstacle_max=dynamic_obstacle_max,
        dynamic_obstacle_vel_min=tuple(dynamic_obstacle_vel_min),
        dynamic_obstacle_vel_max=tuple(dynamic_obstacle_vel_max)
    )
    return dynamic_obstacles, obstacles_fov, dynamic_obstacles_obs


#custom
@njit(cache=True)
def get_dynamic_obstacle_custom(
    obstacle_list_len: int,
    dynamic_obstacles_num: int,
    dynamic_obstacle_dim: int,
    state: np.ndarray,
    obstacles_dynamic: np.ndarray,  # [num]
    obstacles_pos: np.ndarray,  # [x, y, theta]
    obstacles_vel: np.ndarray,  # [vx, omega]
    obstacles_fov: np.ndarray,  # [num]
    dynamic_obstacle_min: float,
    dynamic_obstacle_max: float,
    dynamic_obstacle_vel_min: tuple,
    dynamic_obstacle_vel_max: tuple
) -> np.ndarray:
    VALUE = 255.0
    num = obstacle_list_len if obstacle_list_len > dynamic_obstacles_num else dynamic_obstacles_num
    dynamic_obstacles = np.zeros(
        (num, dynamic_obstacle_dim), dtype=np.float32)
    dynamic_obstacles[:, 0] = VALUE
    count = 0
    for i in range(obstacle_list_len):
        if obstacles_dynamic[i]:  # if the obstacle is dynamic
            dynamic_obs_pos_polar = calc_local_polar(
                obstacles_pos[i], state)  # [r, theta] local
            # normalize
            dynamic_obs_pos_polar = normalize_polar(
                dynamic_obs_pos_polar, dynamic_obstacle_min, dynamic_obstacle_max)
            obstacles_vel[i, 0] = normalize_value(
                np.array([obstacles_vel[i, 0]]), min_value=dynamic_obstacle_vel_min[0], max_value=dynamic_obstacle_vel_max[0])[0]
            obstacles_vel[i, 1] = normalize_value(
                np.array([obstacles_vel[i, 1]]), min_value=dynamic_obstacle_vel_min[1], max_value=dynamic_obstacle_vel_max[1], out_min=-1.0, out_max=1.0)[0]
            dynamic_obstacles[count] = np.concatenate(
                (dynamic_obs_pos_polar, obstacles_vel[i]))
            count += 1
    sort_index = np.argsort(dynamic_obstacles[:, 0])
    dynamic_obstacles_obs = dynamic_obstacles
    dynamic_obstacles = dynamic_obstacles[sort_index]  # sort by distance
    obstacles_fov = obstacles_fov[sort_index]
    condition = (dynamic_obstacles[:, 0] == VALUE)
    dynamic_obstacles[condition, 0] = 0  # replace VALUE with 0
    # select only the first dynamic_obstacles_num obstacles
    dynamic_obstacles = dynamic_obstacles[0:dynamic_obstacles_num]
    return dynamic_obstacles, obstacles_fov[0:dynamic_obstacles_num], dynamic_obstacles_obs[0:dynamic_obstacles_num]

def get_dynamic_obstacles_custom(
    env,
    obstacle_list: list,
    dynamic_obstacles_num: int,
    dynamic_obstacle_dim: int,
    state: np.ndarray,
    dynamic_obstacle_min: float,
    dynamic_obstacle_max: float,
    dynamic_obstacle_vel_min: tuple,
    dynamic_obstacle_vel_max: tuple
) -> np.ndarray:
    obstacle_list_len = len(obstacle_list)
    obstacles_dynamic = np.zeros((obstacle_list_len), dtype=np.bool_)
    obstacles_pos = np.zeros((obstacle_list_len, 2), dtype=np.float32)
    obstacles_vel = np.zeros((obstacle_list_len, 2), dtype=np.float32)
    obstacles_fov = np.zeros((obstacle_list_len), dtype=np.bool_)
    for i in range(obstacle_list_len):
        obs = obstacle_list[i]
        obs_info = obs.get_obstacle_info()
        obstacles_dynamic[i] = obs.get_info().kinematics is not None
        obstacles_pos[i] = obs_info.center.T[0]
        obstacles_vel[i] = obs_info.velocity.T[0]
        obstacles_fov[i] = False
        if obs.get_info().kinematics is not None:
            # print(f"{obs.get_fov_detected_objects()}")
            obstacles_fov[i] = obs.fov_detect_object(env.robot)
    dynamic_obstacles, obstacles_fov, dynamic_obstacles_obs = get_dynamic_obstacle_custom(
        obstacle_list_len=obstacle_list_len,
        dynamic_obstacles_num=dynamic_obstacles_num,
        dynamic_obstacle_dim=dynamic_obstacle_dim,
        state=state,
        obstacles_dynamic=obstacles_dynamic,
        obstacles_pos=obstacles_pos,
        obstacles_vel=obstacles_vel,
        obstacles_fov=obstacles_fov,
        dynamic_obstacle_min=dynamic_obstacle_min,
        dynamic_obstacle_max=dynamic_obstacle_max,
        dynamic_obstacle_vel_min=tuple(dynamic_obstacle_vel_min),
        dynamic_obstacle_vel_max=tuple(dynamic_obstacle_vel_max)
    )
    return dynamic_obstacles, obstacles_fov, dynamic_obstacles_obs

# #custom
# @njit(cache=True)
# def get_dynamic_obstacle_v2(
#     obstacle_list_len: int,
#     dynamic_obstacles_num: int,
#     dynamic_obstacle_dim: int,
#     state: np.ndarray,
#     obstacles_dynamic: np.ndarray,  # [num]
#     obstacles_pos: np.ndarray,  # [x, y, theta]
#     obstacles_vel: np.ndarray,  # [vx, omega]
#     obstacles_fov: np.ndarray,  # [num]
#     dynamic_obstacle_min: float,
#     dynamic_obstacle_max: float,
#     dynamic_obstacle_vel_min: tuple,
#     dynamic_obstacle_vel_max: tuple
# ) -> np.ndarray:
#     VALUE = 255.0
#     num = obstacle_list_len if obstacle_list_len > dynamic_obstacles_num else dynamic_obstacles_num
#     dynamic_obs_poses = np.zeros(
#         (num, 2), dtype=np.float32)
#     dynamic_obs_vels = np.zeros(
#         (num, 2), dtype=np.float32)
#     dynamic_obstacles = np.zeros(
#         (num, dynamic_obstacle_dim), dtype=np.float32)
#     dynamic_obstacles[:, 0] = VALUE
#     count = 0
#     for i in range(obstacle_list_len):
#         if obstacles_dynamic[i]:  # if the obstacle is dynamic
#             dynamic_obs_pos_polar = calc_local_polar(
#                 obstacles_pos[i], state)  # [r, theta] local
#             # normalize
#             dynamic_obs_pos_polar = normalize_polar(
#                 dynamic_obs_pos_polar, dynamic_obstacle_min, dynamic_obstacle_max)
#             obstacles_vel[i, 0] = normalize_value(
#                 np.array([obstacles_vel[i, 0]]), min_value=dynamic_obstacle_vel_min[0], max_value=dynamic_obstacle_vel_max[0])[0]
#             obstacles_vel[i, 1] = normalize_value(
#                 np.array([obstacles_vel[i, 1]]), min_value=dynamic_obstacle_vel_min[1], max_value=dynamic_obstacle_vel_max[1], out_min=-1.0, out_max=1.0)[0]
#             dynamic_obs_poses[count] = dynamic_obs_pos_polar
#             dynamic_obs_vels[count] = obstacles_vel[i]
#             # dynamic_obstacles[count] = np.concatenate(
#             #     (dynamic_obs_pos_polar, obstacles_vel[i]))
#             count += 1
#     sort_index = np.argsort(dynamic_obs_poses[:, 0])
#     dynamic_obstacles_obs = np.concatenate((dynamic_obs_poses, dynamic_obs_vels), axis=1)
#     dynamic_obs_poses = dynamic_obs_poses[sort_index]
#     dynamic_obs_vels = dynamic_obs_vels[sort_index]
#     dynamic_obstacles = np.concatenate((dynamic_obs_poses, dynamic_obs_vels), axis=1)
#     # dynamic_obstacles = dynamic_obstacles[sort_index]  # sort by distance
#     obstacles_fov = obstacles_fov[sort_index]
#     condition = (dynamic_obstacles[:, 0] == VALUE)
#     dynamic_obstacles[condition, 0] = 0  # replace VALUE with 0
#     # select only the first dynamic_obstacles_num obstacles
#     dynamic_obstacles = dynamic_obstacles[0:dynamic_obstacles_num]
#     return dynamic_obstacles, obstacles_fov[0:dynamic_obstacles_num], dynamic_obstacles_obs[0:dynamic_obstacles_num]

# def get_dynamic_obstacles_v2(
#     env,
#     obstacle_list: list,
#     dynamic_obstacles_num: int,
#     dynamic_obstacle_dim: int,
#     state: np.ndarray,
#     dynamic_obstacle_min: float,
#     dynamic_obstacle_max: float,
#     dynamic_obstacle_vel_min: tuple,
#     dynamic_obstacle_vel_max: tuple
# ) -> np.ndarray:
#     obstacle_list_len = len(obstacle_list)
#     obstacles_dynamic = np.zeros((obstacle_list_len), dtype=np.bool_)
#     obstacles_pos = np.zeros((obstacle_list_len, 2), dtype=np.float32)
#     obstacles_vel = np.zeros((obstacle_list_len, 2), dtype=np.float32)
#     obstacles_fov = np.zeros((obstacle_list_len), dtype=np.bool_)
#     for i in range(obstacle_list_len):
#         obs = obstacle_list[i]
#         obs_info = obs.get_obstacle_info()
#         obstacles_dynamic[i] = obs.get_info().kinematics is not None
#         obstacles_pos[i] = obs_info.center.T[0]
#         obstacles_vel[i] = obs_info.velocity.T[0]
#         obstacles_fov[i] = False
#         if obs.get_info().kinematics is not None:
#             # print(f"{obs.get_fov_detected_objects()}")
#             obstacles_fov[i] = obs.fov_detect_object(env.robot)
#     dynamic_obstacles, obstacles_fov, dynamic_obstacles_obs = get_dynamic_obstacle_custom(
#         obstacle_list_len=obstacle_list_len,
#         dynamic_obstacles_num=dynamic_obstacles_num,
#         dynamic_obstacle_dim=dynamic_obstacle_dim,
#         state=state,
#         obstacles_dynamic=obstacles_dynamic,
#         obstacles_pos=obstacles_pos,
#         obstacles_vel=obstacles_vel,
#         obstacles_fov=obstacles_fov,
#         dynamic_obstacle_min=dynamic_obstacle_min,
#         dynamic_obstacle_max=dynamic_obstacle_max,
#         dynamic_obstacle_vel_min=tuple(dynamic_obstacle_vel_min),
#         dynamic_obstacle_vel_max=tuple(dynamic_obstacle_vel_max)
#     )
#     return dynamic_obstacles, obstacles_fov, dynamic_obstacles_obs

#custom
@njit(cache=True)
def get_dynamic_obstacle_v4(
    obstacle_list_len: int,
    dynamic_obstacles_num: int,
    dynamic_obstacle_dim: int,
    state: np.ndarray,
    obstacles_dynamic: np.ndarray,  # [num]
    obstacles_pos: np.ndarray,  # [x, y, theta]
    obstacles_vel: np.ndarray,  # [vx, omega]
    obstacles_fov: np.ndarray,  # [num]
    dynamic_obstacle_min: float,
    dynamic_obstacle_max: float,
    dynamic_obstacle_vel_min: tuple,
    dynamic_obstacle_vel_max: tuple
) -> np.ndarray:
    VALUE = 255.0
    num = obstacle_list_len if obstacle_list_len > dynamic_obstacles_num else dynamic_obstacles_num
    dynamic_obs_poses = np.zeros(
        (num, 3), dtype=np.float32)
    dynamic_obs_vels = np.zeros(
        (num, 2), dtype=np.float32)
    dynamic_obstacles = np.zeros(
        (num, dynamic_obstacle_dim), dtype=np.float32)
    dynamic_obstacles[:, 0] = VALUE
    count = 0
    for i in range(obstacle_list_len):
        if obstacles_dynamic[i]:  # if the obstacle is dynamic
            _dynamic_obs_pos_polar = calc_local_polar(
                obstacles_pos[i], state)  # [r, theta] local
            dynamic_obs_pos = np.array([_dynamic_obs_pos_polar[0],np.cos(_dynamic_obs_pos_polar[1]),np.sin(_dynamic_obs_pos_polar[1])], dtype=np.float32)
            # normalize
            # dynamic_obs_pos_polar = normalize_polar(
            #     dynamic_obs_pos_polar, dynamic_obstacle_min, dynamic_obstacle_max)
            dynamic_obs_pos[0] = normalize_value(
                np.array([dynamic_obs_pos[0]]), min_value=dynamic_obstacle_min, max_value=dynamic_obstacle_max)[0]
            obstacles_vel[i, 0] = normalize_value(
                np.array([obstacles_vel[i, 0]]), min_value=dynamic_obstacle_vel_min[0], max_value=dynamic_obstacle_vel_max[0])[0]
            obstacles_vel[i, 1] = normalize_value(
                np.array([obstacles_vel[i, 1]]), min_value=dynamic_obstacle_vel_min[1], max_value=dynamic_obstacle_vel_max[1], out_min=-1.0, out_max=1.0)[0]
            dynamic_obs_poses[count] = dynamic_obs_pos #dynamic_obs_pos_polar
            dynamic_obs_vels[count] = obstacles_vel[i]
            # dynamic_obstacles[count] = np.concatenate(
            #     (dynamic_obs_pos_polar, obstacles_vel[i]))
            count += 1
    sort_index = np.argsort(dynamic_obs_poses[:, 0])
    dynamic_obs_poses = dynamic_obs_poses[sort_index]
    dynamic_obs_vels = dynamic_obs_vels[sort_index]
    dynamic_obstacles = np.concatenate((dynamic_obs_poses, dynamic_obs_vels), axis=1)
    # dynamic_obstacles = dynamic_obstacles[sort_index]  # sort by distance
    obstacles_fov = obstacles_fov[sort_index]
    condition = (dynamic_obstacles[:, 0] == VALUE)
    dynamic_obstacles[condition, 0] = 0  # replace VALUE with 0
    # select only the first dynamic_obstacles_num obstacles
    dynamic_obstacles = dynamic_obstacles[0:dynamic_obstacles_num]
    return dynamic_obstacles, obstacles_fov[0:dynamic_obstacles_num]

def get_dynamic_obstacles_v4(
    env,
    obstacle_list: list,
    dynamic_obstacles_num: int,
    dynamic_obstacle_dim: int,
    state: np.ndarray,
    dynamic_obstacle_min: float,
    dynamic_obstacle_max: float,
    dynamic_obstacle_vel_min: tuple,
    dynamic_obstacle_vel_max: tuple
) -> np.ndarray:
    obstacle_list_len = len(obstacle_list)
    obstacles_dynamic = np.zeros((obstacle_list_len), dtype=np.bool_)
    obstacles_pos = np.zeros((obstacle_list_len, 2), dtype=np.float32)
    obstacles_vel = np.zeros((obstacle_list_len, 2), dtype=np.float32)
    obstacles_fov = np.zeros((obstacle_list_len), dtype=np.bool_)
    for i in range(obstacle_list_len):
        obs = obstacle_list[i]
        obs_info = obs.get_obstacle_info()
        obstacles_dynamic[i] = obs.get_info().kinematics is not None
        obstacles_pos[i] = obs_info.center.T[0]
        obstacles_vel[i] = obs_info.velocity.T[0]
        obstacles_fov[i] = False
        if obs.get_info().kinematics is not None:
            # print(f"{obs.get_fov_detected_objects()}")
            obstacles_fov[i] = obs.fov_detect_object(env.robot)
    dynamic_obstacles, obstacles_fov = get_dynamic_obstacle_v4(
        obstacle_list_len=obstacle_list_len,
        dynamic_obstacles_num=dynamic_obstacles_num,
        dynamic_obstacle_dim=dynamic_obstacle_dim,
        state=state,
        obstacles_dynamic=obstacles_dynamic,
        obstacles_pos=obstacles_pos,
        obstacles_vel=obstacles_vel,
        obstacles_fov=obstacles_fov,
        dynamic_obstacle_min=dynamic_obstacle_min,
        dynamic_obstacle_max=dynamic_obstacle_max,
        dynamic_obstacle_vel_min=tuple(dynamic_obstacle_vel_min),
        dynamic_obstacle_vel_max=tuple(dynamic_obstacle_vel_max)
    )
    return dynamic_obstacles, obstacles_fov