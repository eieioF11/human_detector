import torch
from tensordict.tensordict import TensorDict, NestedKey
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from typing import Any


class Buffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg, pin_memory=False, show_buffer_info=True):
        self.cfg = cfg
        self._device = torch.device("cuda:0")
        self._pin_memory = pin_memory
        self._capacity = min(cfg.buffer_size, cfg.steps)
        if show_buffer_info:
            print(f"Buffer size: {cfg.buffer_size} steps {cfg.steps}")
            print(f"Buffer capacity: {self._capacity}")
        self._debug_info = show_buffer_info
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=False,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def __len__(self) -> int:
        """Return the number of elements in the buffer."""
        return len(self._buffer)

    def __getitem__(self, index: int | torch.Tensor | NestedKey) -> Any:
        """Get item from the buffer."""
        return self._buffer[index]

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=self._pin_memory,
            # pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, tds, debug=True):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = "cuda:0" if 2.5 * total_bytes < mem_free else "cpu"
        if debug:
            print(f"Buffer capacity: {self._capacity:,}")
            print(f"Storage required: {total_bytes/1e9:.2f} GB")
            print(f"Using {storage_device.upper()} memory for storage.")
        self._storage_device = torch.device(storage_device)
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )

    def load(self, td):
        """
        Load a batch of episodes into the buffer. This is useful for loading data from disk,
        and is more efficient than adding episodes one by one.
        """
        num_new_eps = len(td)
        episode_idx = torch.arange(
            self._num_eps, self._num_eps + num_new_eps, dtype=torch.int64
        )
        td["episode"] = episode_idx.unsqueeze(
            -1).expand(-1, td["reward"].shape[1])
        if self._num_eps == 0:
            self._buffer = self._init(td[0], debug=self._debug_info)
        td = td.reshape(td.shape[0] * td.shape[1])
        self._buffer.extend(td)
        self._num_eps += num_new_eps
        return self._num_eps

    def add(self, td):
        """Add an episode to the buffer."""
        td["episode"] = torch.full_like(
            td["reward"], self._num_eps, dtype=torch.int64)
        if self._num_eps == 0:
            self._buffer = self._init(td, debug=self._debug_info)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        td = td.select(
            "obs", "action", "reward", "terminated", "task", strict=False
        ).to(self._device, non_blocking=True)
        obs = td.get("obs").contiguous()
        action = td.get("action")[1:].contiguous()
        reward = td.get("reward")[1:].unsqueeze(-1).contiguous()
        terminated = td.get("terminated", None)
        if terminated is not None:
            terminated = td.get("terminated")[1:].unsqueeze(-1).contiguous()
        else:
            terminated = torch.zeros_like(reward)
        task = td.get("task", None)
        if task is not None:
            task = task[0].contiguous()
        return obs, action, reward, terminated, task

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)

    def state_dict(self) -> dict[str, Any]:
        """Save the buffer to a file."""
        return {
            "_buffer": self._buffer.state_dict(),
            "_num_eps": self._num_eps,
            "_capacity": self._capacity,
            "_storage_device": self._storage_device,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the buffer from a state dict."""
        buffer_state_dict = state_dict["_buffer"]
        self._num_eps = state_dict["_num_eps"]
        self._capacity = state_dict["_capacity"]
        self._storage_device = state_dict["_storage_device"]
        self._batch_size = buffer_state_dict["_batch_size"]
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=False,
        )
        self._buffer = self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )
        self._buffer.load_state_dict(buffer_state_dict)
