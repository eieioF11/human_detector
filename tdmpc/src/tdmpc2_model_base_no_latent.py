import torch
import torch.nn.functional as F
from tensordict import TensorDict

from common import math
from common.layers import api_model_conversion
from common.scale import RunningScale
from common.world_model_v2_no_latent import WorldModel

TDMPC_TYPE = "tdmpc2_model_base_v2_no_latent"

class TDMPC2(torch.nn.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder["robot"].parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                # {
                #     "params": (
                #         self.model._termination.parameters()
                #         if self.cfg.episodic
                #         else []
                #     )
                # },
                # {"params": self.model._Qs.parameters()},
            ],
            lr=self.cfg.lr,
            capturable=True,
        )
        # self.pi_optim = torch.optim.Adam(
        #     self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True
        # )
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(
            cfg.action_dim >= 20
        )  # Heuristic for large action spaces
        # Simplified discount factor for single-task
        self.discount = self._get_discount(self.cfg.episode_length)

        self._prev_action = torch.zeros(self.cfg.horizon,self.cfg.action_dim,
                                        device=self.device)

        self._prev_mean = torch.nn.Buffer(
            torch.zeros(self.cfg.horizon, self.cfg.action_dim,
                        device=self.device)
        )
        if cfg.compile:
            print("Compiling update function with torch.compile...")
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
                episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
                float: Discount factor for the task.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
        )

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
                fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
                fp (str or dict): Filepath or state dict to load.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(
                fp, map_location=torch.get_default_device(), weights_only=False
            )
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)
        return

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if self.cfg.mpc:
            return self.plan(obs, t0=t0, eval_mode=eval_mode).cpu()
        z, h_obs_vel = self.model.encode(obs)
        action, info = self.model.pi(z)
        if eval_mode:
            action = info["mean"]
        return action[0].cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, h_obs_vel):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        # termination = torch.zeros(
        #     self.cfg.num_samples, 1, dtype=torch.float32, device=z.device
        # )
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(
                self.model.reward(z, actions[t]), self.cfg)
            z = self.model.next(z, actions[t], h_obs_vel)
            G = G + discount * reward
            # G = G + discount * (1 - termination) * reward
            # discount_update simplified for single-task
            discount = discount * self.discount
            # if self.cfg.episodic:
            #     termination = torch.clip(
            #         termination + (self.model.termination(z) > 0.5).float(),
            #         max=1.0,
            #     )
        # action, _ = self.model.pi(z)
        return G
        # return G + discount * (1 - termination) * self.model.Q(
        #     z, action, return_type="avg"
        # )

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False):
        """
        Plan a sequence of actions using the learned world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        z, h_obs_vel = self.model.encode(obs)
        if self.cfg.num_pi_trajs > 0:
            # pi_actions = torch.empty(
            #     self.cfg.horizon,
            #     self.cfg.num_pi_trajs,
            #     self.cfg.action_dim,
            #     device=self.device,
            # )
            # prev_actions = self._prev_action.repeat(1,self.cfg.num_pi_trajs, 1)
            prev_actions = self._prev_action.unsqueeze(1).repeat(1,self.cfg.num_pi_trajs,1)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            _h_obs_vel = h_obs_vel.repeat(self.cfg.num_pi_trajs, 1)
            # print(f"self._prev_action: {self.cfg.num_pi_trajs} {self._prev_action.shape} {z.shape} {_z.shape} {prev_actions.shape}")
            for t in range(self.cfg.horizon - 1):
                # pi_actions[t], _ = self.model.pi(_z)
                # _z = self.model.next(_z, pi_actions[t])
                _z = self.model.next(_z, prev_actions[t], _h_obs_vel)
            # pi_actions[-1], _ = self.model.pi(_z)

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        h_obs_vel = h_obs_vel.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(
            self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full(
            (self.cfg.horizon, self.cfg.action_dim),
            self.cfg.max_std,
            dtype=torch.float,
            device=self.device,
        )
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            # actions[:, : self.cfg.num_pi_trajs] = pi_actions
            actions[:, : self.cfg.num_pi_trajs] = prev_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(
                self.cfg.horizon,
                self.cfg.num_samples - self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=std.device,
            )
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample

            # Compute elite actions
            value = self._estimate_value(z, actions, h_obs_vel).nan_to_num(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (
                score.sum(0) + 1e-9
            )
            std = (
                (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(
                    dim=1
                )
                / (score.sum(0) + 1e-9)
            ).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)

        # Select action
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        self._prev_action = actions
        return a.clamp(-1, 1)

    def update_pi(self, zs):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.

        Returns:
                float: Loss of the policy update.
        """
        action, info = self.model.pi(zs)
        qs = self.model.Q(zs, action, return_type="avg", detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(
            len(qs), device=self.device))
        pi_loss = (
            -(self.cfg.entropy_coef *
              info["scaled_entropy"] + qs).mean(dim=(1, 2))
            * rho
        ).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        if self.pi_optim is not None:
            self.pi_optim.step()
            self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict(
            {
                "pi_loss": pi_loss,
                "pi_grad_norm": pi_grad_norm,
                "pi_entropy": info["entropy"],
                "pi_scaled_entropy": info["scaled_entropy"],
                "pi_scale": self.scale.value,
            }
        )
        return info

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                reward (torch.Tensor): Reward at the current time step.
                terminated (torch.Tensor): Termination signal at the current time step.

        Returns:
                torch.Tensor: TD-target.
        """
        action, _ = self.model.pi(next_z)
        # discount_val simplified for single-task
        discount_val = self.discount
        return reward + discount_val * (1 - terminated) * self.model.Q(
            next_z, action, return_type="min", target=True
        )

    def _update(self, obs, action, reward, terminated):
        # Compute targets
        with torch.no_grad():
            next_z, h_obs_vel = self.model.encode(obs[1:])
            # td_targets = self._td_target(next_z, reward, terminated)

        # Prepare for update
        self.model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z, _ = self.model.encode(obs[0])
        zs[0] = z
        
        consistency_loss = 0
        for t, (_action, _next_z, _h_obs_vel) in enumerate(zip(action.unbind(0), next_z.unbind(0), h_obs_vel.unbind(0))):
            z = self.model.next(z, _action, _h_obs_vel)
            z_r, z_h = self.model.slice_latent(z)
            next_z_r, next_z_h = self.model.slice_latent(_next_z)
            r_consistency_loss = (
                F.mse_loss(z_r, next_z_r) * self.cfg.rho**t
            )
            h_consistency_loss = (
                F.mse_loss(z_h, next_z_h) * self.cfg.rho**t
            )
            consistency_loss = (consistency_loss +
                                r_consistency_loss + h_consistency_loss)
            # consistency_loss = (
            #     consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            # )
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        reward_preds = self.model.reward(_zs, action)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind) in enumerate(
            zip(
                reward_preds.unbind(0),
                reward.unbind(0),
            )
        ):
            reward_loss = (
                reward_loss
                + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean()
                * self.cfg.rho**t
            )

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            # + self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Return training statistics
        self.model.eval()
        info = TensorDict(
            {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                # "value_loss": value_loss,
                "total_loss": total_loss,
                "grad_norm": grad_norm,
            }
        )
        return info.detach().mean()

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        obs, action, reward, terminated, _ = buffer.sample()  # task is ignored
        # kwargs removed as task is no longer passed
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated)
