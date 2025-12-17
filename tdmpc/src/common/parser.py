import dataclasses
import re
from typing import Any

from omegaconf import OmegaConf


def cfg_to_dataclass(cfg, frozen=False):
    """
    Converts an OmegaConf config to a dataclass object.
    This prevents graph breaks when used with torch.compile.
    """
    cfg_dict = OmegaConf.to_container(cfg)
    fields = []
    for key, value in cfg_dict.items():
        fields.append(
            (key, Any, dataclasses.field(default_factory=lambda value_=value: value_))
        )
    dataclass_name = "Config"
    dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)

    def get(self, val, default=None):
        return getattr(self, val, default)

    dataclass.get = get
    return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v is None:
                v = True
        except Exception:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except Exception:
            pass

    return cfg_to_dataclass(cfg)
