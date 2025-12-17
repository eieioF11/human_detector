from irsim.lib import register_behavior
from irsim.lib import reciprocal_vel_obs
from irsim.util.util import relative_position, WrapToPi, omni_to_diff
from irsim.lib.behavior.behavior_methods import DiffRVO, DiffDash
import numpy as np
from math import cos, sin
from irsim.config import env_param, world_param


@register_behavior("diff", "rvo_noize")
def beh_diff_rvo(ego_object, external_objects, **kwargs):
    # print(ego_object)
    # print(external_objects)
    print(kwargs)
    rvo_neighbor = [obj.rvo_neighbor_state for obj in external_objects]
    # print("rvo_neighbor:", rvo_neighbor)
    rvo_state = ego_object.rvo_state
    # print("rvo_state:", rvo_state)
    vxmax = kwargs.get("vxmax", 1.5)
    vymax = kwargs.get("vymax", 1.5)
    acce = kwargs.get("acce", 1.0)
    factor = kwargs.get("factor", 1.0)
    mode = kwargs.get("mode", "rvo")
    neighbor_threshold = kwargs.get("neighbor_threshold", 10.0)
    behavior_vel = DiffRVO(rvo_state, rvo_neighbor, vxmax,
                           vymax, acce, factor, mode, neighbor_threshold)
    print("behavior_vel:", behavior_vel.T)
    v_noise_std = kwargs.get("v_noise_std", 0.0)
    w_noise_std = kwargs.get("w_noise_std", 0.0)
    random_noise = np.random.normal(0, v_noise_std)  # Add some noise
    random_noise_angle = np.random.normal(0, w_noise_std)
    behavior_vel[0] += random_noise
    behavior_vel[1] += random_noise_angle
    print("random_noise:", random_noise)
    np.clip(behavior_vel, -vxmax, vxmax, out=behavior_vel)
    return behavior_vel

@register_behavior("diff", "dash-stop")
def beh_diff_dash_stop(ego_object, external_objects, **kwargs):
    state = ego_object.state
    goal = ego_object.goal
    goal_threshold = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    lst = list(filter(lambda x: x.name.find("robot") != -1, external_objects))
    detect = False
    for obj in lst:
        if ego_object.fov_detect_object(obj):
            detect = True
            break

    if goal is None:
        if world_param.count % 10 == 0:
            env_param.logger.warning("Goal is currently None. This dash behavior is waiting for goal configuration")

        return np.zeros((2, 1))

    behavior_vel = DiffDash(state, goal, max_vel, goal_threshold, angle_tolerance)
    if detect:
        behavior_vel = np.array([[0.0], [0.0]])
    return behavior_vel

@register_behavior("diff", "rvo-stop")
def beh_diff_rvo_stop(ego_object, external_objects, **kwargs):
    rvo_neighbor = [obj.rvo_neighbor_state for obj in external_objects]
    rvo_state = ego_object.rvo_state
    vxmax = kwargs.get("vxmax", 1.5)
    vymax = kwargs.get("vymax", 1.5)
    acce = kwargs.get("acce", 1.0)
    factor = kwargs.get("factor", 1.0)
    mode = kwargs.get("mode", "rvo")
    neighbor_threshold = kwargs.get("neighbor_threshold", 10.0)
    behavior_vel = DiffRVO(rvo_state, rvo_neighbor, vxmax,
                           vymax, acce, factor, mode, neighbor_threshold)
    lst = list(filter(lambda x: x.name.find("robot") != -1, external_objects))
    detect = False
    for obj in lst:
        if ego_object.fov_detect_object(obj):
            detect = True
            break
    if detect:
        behavior_vel = np.array([[0.0], [0.0]])
    return behavior_vel

@register_behavior("diff", "rvo-dash")
def beh_diff_rvo_dash(ego_object, external_objects, **kwargs):
    rvo_neighbor = [obj.rvo_neighbor_state for obj in external_objects]
    rvo_state = ego_object.rvo_state
    l_vxmax = kwargs.get("low_vxmax", 1.0)
    l_vymax = kwargs.get("low_vymax", 1.0)
    f_vxmax = kwargs.get("fast_vxmax", 1.5)
    f_vymax = kwargs.get("fast_vymax", 1.5)
    acce = kwargs.get("acce", 1.0)
    factor = kwargs.get("factor", 1.0)
    mode = kwargs.get("mode", "rvo")
    neighbor_threshold = kwargs.get("neighbor_threshold", 10.0)
    lst = list(filter(lambda x: x.name.find("robot") != -1, external_objects))
    detect = False
    for obj in lst:
        if ego_object.fov_detect_object(obj):
            detect = True
            break
    vxmax = l_vxmax
    vymax = l_vymax
    if detect:
        vxmax = f_vxmax
        vymax = f_vymax
    behavior_vel = DiffRVO(rvo_state, rvo_neighbor, vxmax,
                           vymax, acce, factor, mode, neighbor_threshold)
    return behavior_vel

@register_behavior("diff", "sfm")
def beh_diff_sfm(ego_object, external_objects, **kwargs):
    """
    Social Force Model (SFM) / Artificial Potential Field (APF) に基づく
    カスタムビヘイビア。

    ワールド座標系で引力・斥力を計算し、ロボットのローカルな
    速度指令 [v, w] に変換して返します。
    """

    # --- 1. パラメータと状態の取得 ---
    # SFMパラメータ
    k_goal = kwargs.get("k_goal", 1.0)
    k_agent = kwargs.get("k_agent", 1.5)
    sigma_agent = kwargs.get("sigma_agent", 1.0)
    repel_range = kwargs.get("repel_range", 5.0)
    # [v, w] 変換用パラメータ
    k_v = kwargs.get("k_v", 1.0) # 角度差による速度減衰ゲイン
    k_w = kwargs.get("k_w", 2.0) # 角速度ゲイン

    # ロボットの最大速度 [v_max, w_max] を取得
    min_arr, max_arr = ego_object.get_vel_range()
    vmin = min_arr[0]
    wmin = min_arr[1]
    vmax = max_arr[0]
    wmax = max_arr[1]

    # ロボットの現在の状態 [x, y, theta]
    # .flatten() で (3,) の形状に統一
    # (ここでエラーが出ていない＝環境設定の問題)
    ego_state = ego_object.state.flatten()
    ego_pos = ego_state[0:2]      # 現在位置 [x, y]
    current_theta = ego_state[2]  # 現在の角度 theta

    # --- 2. 目標への引力 (vel_goal) [ワールド座標系] ---
    vel_goal = np.array([0.0, 0.0])
    goal_reached = False

    if ego_object.goal is not None:
        goal_pos = ego_object.goal[0:2].flatten() # (2,) に統一
        vec_to_goal = goal_pos - ego_pos
        dist_to_goal = np.linalg.norm(vec_to_goal)

        if dist_to_goal < ego_object.goal_threshold:
            goal_reached = True # 目標に到達
        else:
            dir_to_goal = vec_to_goal / dist_to_goal
            vel_goal = k_goal * dir_to_goal 

    # --- 3. 他エージェントからの斥力 (vel_repel) [ワールド座標系] ---
    vel_repel = np.array([0.0, 0.0])
    for obj in external_objects:
        agent_pos = obj.state[0:2].flatten() # (2,) に統一
        vec_from_agent = ego_pos - agent_pos
        dist_from_agent = np.linalg.norm(vec_from_agent)
        if dist_from_agent > repel_range or dist_from_agent < 1e-6:
            continue
        dir_from_agent = vec_from_agent / dist_from_agent
        magnitude = k_agent * np.exp(-dist_from_agent / sigma_agent)
        vel_repel += magnitude * dir_from_agent
    # --- 4. ワールド速度の合成 ---

    desired_vel_world = vel_goal + vel_repel
    speed_world = np.linalg.norm(desired_vel_world)

    # --- 5. ローカル速度 [v, w] への変換 ---

    if goal_reached or speed_world < 1e-3:
        return np.array([[0.0], [0.0]]) # [v, w] = [0, 0]

    desired_angle = np.arctan2(desired_vel_world[1], desired_vel_world[0])
    angle_diff = desired_angle - current_theta
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    w = k_w * angle_diff
    w = np.clip(w, -wmax, wmax)

    v = speed_world * k_v * np.cos(angle_diff)
    v = np.clip(v, 0, vmax) # 0 (後退なし) から最大速度でクリップ
    # --- 6. 速度指令を (2, 1) の列ベクトルで返す ---
    return np.array([v, w])