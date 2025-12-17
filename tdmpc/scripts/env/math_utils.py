import math
import numpy as np
import random
from numba import njit
import torch


PI = np.pi
TWO_PI = 2 * np.pi
FLOAT32_MAX = np.finfo(np.float32).max
FLOAT32_MIN = np.finfo(np.float32).min


@njit(cache=True)
def normalize_value(value: np.ndarray, min_value: float, max_value: float, out_min: float = 0.0, out_max: float = 1.0) -> np.ndarray:
    """
    Normalize a value to the range [out_min, out_max].
    Args:
        value (np.ndarray): The value to normalize.
        min_value (float): The minimum value of the range.
        max_value (float): The maximum value of the range.
    Returns:
        np.ndarray: Normalized value in the range [out_min, out_max].
    """
    value = np.clip(value, min_value, max_value)
    return ((value.astype(np.float32) - min_value) / (max_value - min_value) * (out_max - out_min) + out_min).astype(np.float32)

@njit(cache=True)
def normalize_polar(polar: np.ndarray, distance_min: float, distance_max: float) -> np.ndarray:
    """
    Normalize polar coordinates (r, theta) to a specific range.
    Args:
        polar (np.ndarray): Input array of shape (2,) representing (r, theta) polar coordinates.
        distance_min (float): Minimum value for the distance (r).
        distance_max (float): Maximum value for the distance (r).
    Returns:
        np.ndarray: Normalized polar coordinates in the format (normalized_r, normalized_theta).
    """
    polar = polar.astype(np.float32)
    # Normalize the distance (r) to the range [distance_min, distance_max]
    polar[0] = normalize_value(
        np.array([polar[0]]), min_value=distance_min, max_value=distance_max
    )[0]
    # and the angle (theta) to the range [-pi, pi]
    polar[1] = normalize_value(np.array([polar[1]]), min_value=-np.pi, max_value=np.pi, out_min=-1.0, out_max=1.0)[0]
    return polar

@njit(cache=True)
def normalize_angle(angle: np.ndarray) -> np.ndarray:
    """Normalize angle to be in the range [-pi, pi]."""
    a = ((angle+PI) % TWO_PI) - PI
    two_pis = np.array([TWO_PI]*angle.shape[0])
    a += np.where(a < -PI, two_pis,
                  np.zeros(angle.shape[0]))
    return a


@njit(cache=True)
def transfom_robot(map_x: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
    """
    Transform map to robot coordinates.
    Args:
        map_x (np.ndarray): (x, y, theta)
        robot_state (np.ndarray): (x, y, theta)
    Returns:
        np.ndarray: Transformed coordinates in the format (x, y, theta).
    """
    # robot pose
    rx, ry, rtheta = robot_state
    # transform
    dx = map_x[0] - rx
    dy = map_x[1] - ry
    dtheta = normalize_angle(
        np.array([map_x[2] - rtheta], dtype=np.float32))[0]
    # rotation (global -> local)
    x_r = np.cos(-rtheta) * dx - np.sin(-rtheta) * dy
    y_r = np.sin(-rtheta) * dx + np.cos(-rtheta) * dy
    theta_r = dtheta
    return np.array([x_r, y_r, theta_r], dtype=np.float32)


@njit(cache=True)
def calc_polar_coord(x: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).
    Args:
        x (np.ndarray): Input array of shape (2,) representing (x, y) coordinates.
    Returns:
        np.ndarray: Output array of shape (2,) representing (r, theta) polar coordinates.
    """
    r = np.linalg.norm(x.astype(np.float32))
    theta = np.arctan2(x[1], x[0])
    return np.array([r, theta], dtype=np.float32)

@njit(cache=True)
def calc_local_polar(target: np.ndarray, state: np.ndarray) -> np.ndarray:
    diff = (target[:2] - state[:2]).astype(np.float32)
    r = np.linalg.norm(diff)
    theta = np.arctan2(diff[1], diff[0])
    theta = normalize_angle(
        np.array([theta-state[2]], dtype=np.float32))[0]
    return np.array([r, theta], dtype=np.float32)

@njit(cache=True)
def calc_cossin(state: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the cosine and sine of the angle between two 2D vectors.
    Args:
        vec1 (list): First 2D vector.
        vec2 (list): Second 2D vector.
    Returns:
        (tuple): (cosine, sine) of the angle between the vectors.
    """
    vec1 = np.array([np.cos(state[2]).item(), np.sin(
        state[2]).item()], dtype=np.float32)
    vec2 = np.array([
        goal[0].item() - state[0].item(),
        goal[1].item() - state[1].item(),
    ], dtype=np.float32)

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    cos = np.dot(vec1, vec2)
    sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return cos, sin


@njit(cache=True)
def set_seed(seed: int) -> None:
    """
    Set the random seed for NumPy.
    Args:
        seed (int): The seed value to set.
    """
    np.random.seed(seed)
    random.seed(seed)

def set_all_seeds(seed: int) -> None:
    """
    Set the random seed for NumPy and Python's random module.
    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# @njit(cache=True)
def random_state(x_limit: np.ndarray, y_limit: np.ndarray, theta_limit: np.ndarray = np.array([-PI, PI])) -> np.ndarray:
    """
    Generate a random start position within the specified limits.
    Returns:
        np.ndarray: Random start position in the format [[x], [y], [theta]].
    """
    x = np.random.uniform(x_limit[0], x_limit[1])
    y = np.random.uniform(y_limit[0], y_limit[1])
    theta = np.random.uniform(theta_limit[0], theta_limit[1])
    return np.array([[x], [y], [theta]], dtype=np.float32)

@njit(cache=True)
def make_state(x: float, y: float, theta: float) -> np.ndarray:
    """
    Create a state vector from given x, y, and theta values.
    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        theta (float): Orientation angle in radians.
    Returns:
        np.ndarray: State vector in the format [[x], [y], [theta]].
    """
    return np.array([[x], [y], [theta]], dtype=np.float32)

@njit(cache=True)
def np_norm(x: np.ndarray) -> float:
    """
    Compute the Euclidean norm (L2 norm) of a vector.
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
    Returns:
        float: Euclidean norm between the two vectors.
    """
    return np.linalg.norm((x).astype(np.float32))

@njit(cache=True)
def np_norm(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Euclidean norm (L2 norm) between two vectors.
    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
    Returns:
        float: Euclidean norm between the two vectors.
    """
    return np.linalg.norm((x - y).astype(np.float32))

@njit(cache=True)
def compute_spl(success_list: np.ndarray, path_length_list: np.ndarray, shortest_path_length_list: np.ndarray) -> float:
    """
    Compute the SPL (Success weighted Path Length)

    Args:
        success_list (list of bool or int): 成功フラグ（0または1）
        path_length_list (list of float): 実際に移動した距離（p_i）
        shortest_path_length_list (list of float): 最短経路長（l_i）

    Returns:
        float: SPL値（0〜1の間）
    """
    assert len(success_list) == len(path_length_list) == len(shortest_path_length_list), \
        "すべてのリストの長さが同じである必要があります"

    N = len(success_list)
    spl_sum = 0.0

    for s, p, l in zip(success_list, path_length_list, shortest_path_length_list):
        if s:
            spl_sum += l / max(p, l)

    return spl_sum / N

@njit(cache=True)
def approx_zero(value: float,limit: float = 1e-6) -> bool:
    """Approximate zero for numerical stability."""
    if abs(value) < limit:
        return True
    return False