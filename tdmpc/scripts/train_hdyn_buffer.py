import os
import warnings
from pathlib import Path
from time import time
import math
import random
import sys
import re

import hydra
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from termcolor import colored
from omegaconf import OmegaConf
from logging import getLogger

import env.register_env
from env.make_env import make_nav_env
from env.navigation_env_cfg import NavigationEnvCfg, set_navigation_env_cfg, cfg_path
from env.math_utils import random_state

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed

# Custom
# from tdmpc2_model_base_hdyn import TDMPC2, TDMPC_TYPE
# from tdmpc2_model_base_hdyn_rnn import TDMPC2, TDMPC_TYPE
from tdmpc2_model_base_hdyn_no_latent import TDMPC2, TDMPC_TYPE

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_CUDA_LOG_LEVEL"] = "ERROR"

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="torch._logging")
warnings.filterwarnings("ignore", module="torch._dynamo")


def common_metrics_fn(_step: int, _ep_idx: int, _start_time: float) -> dict:
    """Return a dictionary of current metrics."""
    elapsed_time = time() - _start_time
    return dict(
        step=_step,
        episode=_ep_idx,
        elapsed_time=elapsed_time,
        steps_per_second=(
            _step / elapsed_time if elapsed_time > 0 else 0.0
        ),  # Avoid division by zero
    )

def to_td_fn(env, obs, action=None, reward=None, terminated=None) -> TensorDict:
    """Creates a TensorDict for a new episode."""
    # print("obs:", obs)
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device="cpu")
    else:
        obs = obs.unsqueeze(0).cpu()
    # print("obs_tensor:", obs)
    if action is None:
        action = torch.full_like(env.rand_act(), float("nan"))
    if reward is None:
        reward = torch.tensor(float("nan"))
    if terminated is None:
        terminated = torch.tensor(float("nan"))
    # print("action:", action.unsqueeze(0))
    td = TensorDict(
        obs=obs,
        action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
        terminated=terminated.unsqueeze(0),
        batch_size=(1,),
    )
    # print("td:", td)
    return td


@hydra.main(config_name="hdyn_train_buffer_model_parameter.yaml", config_path=cfg_path)
def train(cfg: dict):
    """
    Script for training single-task using TD-MPC2
    """
    # Set up logging
    logger = getLogger(__name__)
    logger.info('Starting environment test script')
    logger.debug(f'cuda available: {torch.cuda.is_available()}')
    model_cfg = OmegaConf.to_container(cfg)
    config = OmegaConf.load(os.path.join(
        cfg_path, "env", cfg.env_parameter_file))
    merged_cfg = OmegaConf.merge(model_cfg, config)
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."

    # Set up the environment
    # cfg = parse_cfg(cfg)
    cfg = parse_cfg(merged_cfg)
    DEBUG_LEVEL = cfg.debug_level
    logger.setLevel(DEBUG_LEVEL)
    logger.info(f"debug level: {DEBUG_LEVEL}")
    env_cfg = NavigationEnvCfg()
    set_navigation_env_cfg(env_cfg, config)
    set_seed(cfg.seed)
    logger.info(f"seed: {cfg.seed}")
    env_cfg.logger = logger
    env_cfg.seed = cfg.seed
    env_cfg.episode_length_s = cfg.episode_length_s
    env_cfg.max_episode_length = math.ceil(
        env_cfg.episode_length_s / env_cfg.dt)
    if (not env_cfg.display):
        cfg.save_video = False
    cfg.world["dt"] = env_cfg.dt
    logger.info(f"save_video: {cfg.save_video}")
    cfg.episode_length = env_cfg.max_episode_length
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    cfg.num_envs = env_cfg.num_envs
    cfg.num_pedestrians = re.findall(r'\d+', cfg.irsim['config_file'])[0]
    logger.info(f"num_pedestrians: {cfg.num_pedestrians}")
    logger.info(f"num_envs: {cfg.num_envs}")
    cfg.fine_tuning = True
    model_mode = ""
    if cfg.latent_separate:
        model_mode = "separation"
        cfg.latent_dim = cfg.latent_r_dim + cfg.latent_h_dim
        logger.info(
            f"latent_dim: {cfg.latent_dim} h_dim: {cfg.latent_h_dim} r_dim: {cfg.latent_r_dim}")
    else:
        model_mode = "normal"
        cfg.latent_h_dim = "??"
        cfg.latent_r_dim = "??"
        logger.info(f"latent_dim: {cfg.latent_dim}")
    # make navigation environment
    env = make_nav_env(cfg, env_cfg, env_id=cfg.env_id, use_dt_sep=True)
    logger.info("obs_shape {} r_obs_shape: {}, h_obs_shape: {}".format(
        cfg.obs_shape, cfg.r_obs_shape, cfg.h_obs_shape))

    # set variables of config
    cfg.work_dir = (
        Path(hydra.utils.get_original_cwd())
        / "logs"
        / cfg.task
        / ("seed_" + str(cfg.seed))
        / cfg.exp_name
    )
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (
        cfg.num_bins - 1
    )  # Bin size for discrete regression

    logger.info(colored("Work dir:", "yellow",
                attrs=["bold"]) + str(cfg.work_dir))
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger_instance = Logger(cfg, defined_metric=True, tdmpc_type=TDMPC_TYPE)

    # Load agent
    ws = os.path.join(hydra.utils.get_original_cwd(), "models")

    buffer_path = os.path.join(
        ws, cfg.buffer_dir, str(cfg.num_pedestrians), model_mode, cfg.buffer_name)
    assert os.path.exists(buffer_path), f"buffer_path {buffer_path} not found! Must be a valid filepath."

    logger.info(f"Loading buffer from {buffer_path}...")
    state_dict = torch.load(
        buffer_path, map_location=torch.get_default_device(), weights_only=False
    )
    buffer.load_state_dict(state_dict)
    logger.info("Buffer loaded.")

    # Variables from OnlineTrainer.__init__
    _step: int = 0
    _ep_idx: int = 0
    _start_time: float = time()

    # Main training loop from OnlineTrainer.train
    train_metrics = {}
    logger.info(colored("Starting training...", "green", attrs=["bold"]))
    while _step <= cfg.steps:

        if _step > 0:
            current_train_metrics = {}  # Renamed to avoid conflict
            current_train_metrics.update(
                common_metrics_fn(_step, _ep_idx, _start_time)
            )
            current_train_metrics.update(train_metrics)  # logにlossなどの情報追加
            logger_instance.log(current_train_metrics, "train")
        print(
            colored(f"train step {_step} / {cfg.steps}", "blue", attrs=["bold"]), end="\r")

        # Update agent
        if _step >= cfg.seed_steps:
            if _step == cfg.seed_steps:
                num_updates = cfg.seed_steps
                print("")
                logger.info("Pre-training agent on seed data...")
            else:
                num_updates = 1
            for _ in range(num_updates):
                _agent_update_metrics = agent.update(buffer)

            train_metrics.update(_agent_update_metrics)

        _step += 1

    logger_instance.finish(agent, identifier=f"{cfg.seed}")
    logger.info("\nTraining completed successfully")


if __name__ == "__main__":
    train()
