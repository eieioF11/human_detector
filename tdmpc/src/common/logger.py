import dataclasses
import datetime
import os
import re

import numpy as np
import pandas as pd
from termcolor import colored

CONSOLE_FORMAT = [
    ("iteration", "Iteration", "int"),
    ("episode", "Episode", "int"),
    ("step", "Step", "int"),
    ("env_id", "Env ID", "int"),
    ("episode_reward", "Reward", "float"),
    ("episode_success", "Success", "float"),
    ("elapsed_time", "Time", "time"),
    ("episode_collision", "Collision Rate", "float"),
    ("episode_mileage", "Mileage", "float"),
    ("episode_shortest_length", "Shortest Path Length", "float"),
    ("episode_length", "Episode Length", "float"),
    ("episode_spl", "SPL", "float"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "train_loss": "blue",
    "eval": "green",
    "eval_results": "green",
    "env_eval": "green",
    "env_eval_results": "green",
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """
    Pretty-printing of current run information.
    Logger calls this method at initialization.
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs),
            _limstr(v),
        )

    observations = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("steps", f"{int(cfg.steps):,}"),
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """
    Return a wandb-safe group name for logging.
    Optionally returns group name as list.
    """
    lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, cfg, wandb):
        self.cfg = cfg
        self._save_dir = make_dir(cfg.work_dir / "eval_video")
        self._wandb = wandb
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self._save_dir and self._wandb and enabled
        # print("enabled: ",self.enabled)
        self.fps = env.metadata.get("render_fps", 30)
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render()
            # print(type(frame))
            if isinstance(frame, list):
                for f in frame:
                    self.frames.append(f)
            else:
                self.frames.append(frame)

    def save(self, step, key="videos/eval_video"):
        if self.enabled and len(self.frames) > 0:
            frames = np.stack(self.frames)
            return self._wandb.log(
                {
                    key: self._wandb.Video(
                        frames.transpose(0, 3, 1, 2), fps=self.fps, format="mp4"
                    )
                },
                step=step,
            )


class Logger:
    """Primary logging object. Logs either locally or using wandb."""

    def __init__(self, cfg, name=None, defined_metric=False, tdmpc_type="tdmpc"):
        self._log_dir = make_dir(cfg.work_dir)
        latent_separate = ""
        ls = "normal"
        if cfg.latent_separate:
            latent_separate = "-latent_separate"
            ls = "separation"
        env_name = cfg.irsim['config_file'].replace('.yaml', '')
        print(f"env:{env_name}")
        ped_mode = "vo"
        if env_name.find("rvo") != -1:
            ped_mode = "rvo"
        if env_name.find("hrvo") != -1:
            ped_mode = "hrvo"
        if env_name.find("dash") != -1:
            ped_mode = "dash"
        if env_name.find("sfm") != -1:
            ped_mode = "sfm"
        if env_name.find("dash-stop") != -1:
            ped_mode = "dash-stop"
        if env_name.find("rvo-stop") != -1:
            ped_mode = "rvo-stop"
        print(f"ped_mode:{ped_mode}")
        # self._date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self._date}_{ped_mode}_{cfg.num_pedestrians}_{ls}"
        self._model_dir = make_dir(self._log_dir / "models" / dir_name)
        self._save_csv = cfg.save_csv
        self._save_agent = cfg.save_agent
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = []
        print_run(cfg)
        self.project = cfg.get("wandb_project", "none")
        self.entity = cfg.get("wandb_entity", "none")
        if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            self._video = None
            return
        os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
        import wandb
        # set wandb settings
        tags = cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"] + [f"num_envs:{cfg.num_envs}"] + [
            f"steps:{cfg.steps}"] + [f"ep_len:{cfg.episode_length}"] + [f"env:{env_name}"] + [
                f"latent_separate:{cfg.latent_separate}"] + [f"latent_r_dim:{cfg.latent_r_dim}"] + [f"latent_h_dim:{cfg.latent_h_dim}"] + [
                    f"num_pedestrians:{cfg.num_pedestrians}"] + [f"ped_mode:{ped_mode}"]
        tags += [f"tdmpc-type:{tdmpc_type}"]
        print(f"wandb tags: {tags}")
        # print(f"num pedestrians: {cfg.num_pedestrians}")
        if name is None:
            name = str(cfg.seed) + \
                f"-{cfg.num_pedestrians}_ped" + \
                latent_separate + "_" + ped_mode
        print(f"wandb name: {name}")

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=str(name),
            # name=str(cfg.seed),
            group=self._group,
            tags=tags,
            dir=self._log_dir,
            config=dataclasses.asdict(cfg),
        )
        print(colored("Logs will be synced with wandb.",
              "blue", attrs=["bold"]))
        if defined_metric:
            wandb.define_metric("train_loss", step_metric="iteration")
        self._wandb = wandb
        self._video = (
            VideoRecorder(
                cfg, self._wandb) if self._wandb and cfg.save_video else None
        )

    @property
    def video(self):
        return self._video

    @property
    def model_dir(self):
        return self._model_dir

    def save_agent(self, agent=None, identifier="final"):
        if self._save_agent and agent:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + "-" +
                    str(self._seed) + "-" + str(identifier),
                    type="model",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def save_buffer(self, buffer=None, identifier="final"):
        if self._save_agent and buffer:
            import torch
            fp = self._model_dir / f"buffer_{str(identifier)}.pt"
            torch.save(buffer.state_dict(), fp)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + "-" +
                    str(self._seed) + "-buffer-" + str(identifier),
                    type="buffer",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def finish(self, agent=None, identifier="final"):
        try:
            self.save_agent(agent, identifier)
        except Exception as e:
            print(colored(f"Failed to save model: {e}", "red"))
        if self._wandb:
            self._wandb.finish()

    def finish_wandb(self):
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def log_image(self, img, category="image", step=0):
        if self._wandb:
            self._wandb.log(({category: self._wandb.Image(img)}), step=step)

    def log_bar_chart(self, data, title="Bar Chart", columns=["env_id", "episode_success"]):
        if self._wandb:
            table = self._wandb.Table(data=data, columns=columns)
            self._wandb.log({title: self._wandb.plot.bar(
                table,
                columns[0],
                columns[1],
                title=title,
            )})

    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb:
            if category in {"train", "eval"}:
                xkey = "step"
            elif category in {"env_eval", "env_eval_results"}:
                xkey = "env_id"
            elif category == "pretrain":
                xkey = "iteration"
            _d = dict()
            for k, v in d.items():
                _d[category + "/" + k] = v
            if category in {"eval_results", "train_loss"}:
                self._wandb.log(_d)
            else:
                self._wandb.log(_d, step=d[xkey])
        if category == "eval" and self._save_csv:
            keys = ["step", "episode_reward"]
            self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
            pd.DataFrame(np.array(self._eval)).to_csv(
                self._log_dir / "eval.csv", header=keys, index=None
            )
        self._print(d, category)
