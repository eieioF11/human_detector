from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

from common import init, layers, math


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._action_masks = None  # No action masks needed

        if cfg.latent_separate:
            print("cfg.obs_shape:", cfg.obs_shape)
            self._encoder_r = layers.enc_separate(
                cfg, cfg.latent_r_dim, cfg.r_obs_shape)
            self._dynamics_r = layers.lstm(
                cfg.latent_r_dim + cfg.action_dim,
                cfg.mlp_dim,
                cfg.latent_r_dim
            )
            # self._dynamics_r = layers.mlp(
            #     cfg.latent_r_dim + cfg.action_dim,
            #     2 * [cfg.mlp_dim],
            #     cfg.latent_r_dim,
            #     act=layers.SimNorm(cfg),
            # )
            self._encoder_h = layers.enc_separate(
                cfg, cfg.latent_h_dim, cfg.h_obs_shape)
            # self._dynamics_h = layers.mlp(
            #     cfg.latent_h_dim,
            #     2 * [cfg.mlp_dim],
            #     cfg.latent_h_dim,
            #     act=layers.SimNorm(cfg),
            # )
            self._dynamics_h = layers.lstm(
                cfg.latent_h_dim,
                cfg.mlp_dim,
                cfg.latent_h_dim
            )
            # self._dynamics_h = layers.lstm(
            #     cfg.latent_h_dim + cfg.action_dim,
            #     cfg.mlp_dim,
            #     cfg.latent_h_dim
            # )
            # self._dynamics_h = layers.mlp(
            #     cfg.latent_h_dim + cfg.action_dim,
            #     2 * [cfg.mlp_dim],
            #     cfg.latent_h_dim,
            #     act=layers.SimNorm(cfg),
            # )
            self._encoder = nn.ModuleDict(
                {
                    "robot": self._encoder_r["state"],
                    "human": self._encoder_h["state"],
                }
            )
            self._dynamics = nn.ModuleDict(
                {
                    "robot": self._dynamics_r,
                    "human": self._dynamics_h,
                }
            )
            self._r_obs_s = 0
            self._r_obs_e = self.cfg.r_obs_shape["state"][0]
            self._h_obs_s = self._r_obs_e
            self._h_obs_e = self._h_obs_s + self.cfg.h_obs_shape["state"][0]
            self._latent_r_s = 0
            self._latent_r_e = self.cfg.latent_r_dim
            self._latent_h_s = self._latent_r_e
            self._latent_h_e = self._latent_h_s + self.cfg.latent_h_dim
        else:
            self._encoder = layers.enc(cfg)
            self._dynamics = layers.mlp(
                cfg.latent_dim + cfg.action_dim,
                2 * [cfg.mlp_dim],
                cfg.latent_dim,
                act=layers.SimNorm(cfg),
            )
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._termination = (
            layers.mlp(cfg.latent_dim, 2 *
                       [cfg.mlp_dim], 1) if cfg.episodic else None
        )
        self._pi = layers.mlp(
            cfg.latent_dim,
            2 * [cfg.mlp_dim],
            2 * cfg.action_dim,
        )
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer(
            "log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min
        )
        self.init()

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(
            self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(
            self._Qs.params.data.clone(), no_convert=True
        )

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = "TD-MPC2 World Model\n"
        modules = [
            "Encoder",
            "Dynamics",
            "Reward",
            "Termination",
            "Policy prior",
            "Q-functions",
        ]
        for i, m in enumerate(
            [
                self._encoder,
                self._dynamics,
                self._reward,
                self._termination,
                self._pi,
                self._Qs,
            ]
        ):
            if m == self._termination and not self.cfg.episodic:
                continue
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def slice_observation(self, obs):
        """
        Slices the observation into robot and human parts if using separate latents.
        """
        r_obs = obs[..., self._r_obs_s:self._r_obs_e]
        h_obs = obs[..., self._h_obs_s:self._h_obs_e]
        return r_obs, h_obs

    def slice_latent(self, z):
        z_r = z[..., self._latent_r_s:self._latent_r_e]
        z_h = z[..., self._latent_h_s:self._latent_h_e]
        return z_r, z_h

    def encode(self, obs):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.latent_separate:
            # print("obs shape:", obs.shape)
            # print("obs:", obs)
            r_obs, h_obs = self.slice_observation(obs)
            # print("r_obs:\n", r_obs, "\nh_obs:\n", h_obs)
            z_r = self._encoder["robot"](r_obs)
            z_h = self._encoder["human"](h_obs)
            return torch.cat([z_r, z_h], dim=-1)
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.latent_separate:
            z_r, z_h = self.slice_latent(z)
            z_next_r = self._dynamics["robot"](torch.cat([z_r, a], dim=-1))
            z_next_h = self._dynamics["human"](z_h)
            # z_next_h = self._dynamics["human"](torch.cat([z_h, a], dim=-1))
            return torch.cat([z_next_r, z_next_h], dim=-1)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a):
        """
        Predicts instantaneous (single-step) reward.
        """
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(self, z, unnormalized=False):
        """
        Predicts termination signal.
        """
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """

        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        action_dims = None  # Always single task

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict(
            {
                "mean": mean,
                "log_std": log_std,
                "action_prob": 1.0,
                "entropy": -log_prob,
                "scaled_entropy": -log_prob * entropy_scale,
            }
        )
        return action, info

    def Q(self, z, a, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        # multitask related z processing is removed
        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2
