import os
import pickle
from typing import Dict, Optional, Union

import d4rl
import gym
import numpy as np
import torch

from wiserl.utils import utils

prefix = "datasets/pt_ipl/"
DATASET_PATH={
    "hopper-medium-replay-v2": f"{prefix}/hopper-medium-replay-v2/num500",
    "hopper-medium-expert-v2": f"{prefix}/hopper-medium-expert-v2/num100",
    "walker2d-medium-replay-v2": f"{prefix}/walker2d-medium-replay-v2/num500",
    "walker2d-medium-expert-v2": f"{prefix}/walker2d-medium-expert-v2/num100",
    "hammer-cloned-v1": f"{prefix}/hammer-cloned-v1/num100",
    "hammer-human-v1": f"{prefix}/hammer-human-v1/num100",
    "pen-cloned-v1": f"{prefix}/pen-cloned-v1/num100",
    "pen-human-v1": f"{prefix}/pen-human-v1/num100",
    "Can-mh": f"{prefix}/Can/num500_q100",
    "Can-ph": f"{prefix}/Can/num100_q50",
    "Lift-mh": f"{prefix}/Lift/num500_q100",
    "Lift-ph": f"{prefix}/Lift/num100_q50",
    "Square-mh": f"{prefix}/Square/num500_q100",
    "Square-mh": f"{prefix}/Square/num100_q50",
}


class IPLComparisonOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space,
        action_space,
        env: str,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str = "human",
        reward_scale: float = 0.01,  # only used in soft mode
        eval: bool = False,
    ):
        super().__init__()
        assert env in DATASET_PATH.keys(), f"Env {env} not registered for PT dataset"
        assert mode in {"human", "script", "soft"}, "Supported modes for IPLComparisonOfflineDataset: {human, script, soft}"

        self.env_name = env
        self.mode = mode
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.eval = eval
        self.reward_scale = reward_scale
        train_or_eval = "eval" if eval else "train"
        mode_name = mode if mode != "soft" else "human"
        path = f"{DATASET_PATH[self.env_name]}_{mode_name}_{train_or_eval}.npz"
        with open(path, "rb") as f:
            data = np.load(f)
            data = utils.nest_dict(data)
            if capacity is not None:
                data = utils.get_from_batch(data, 0, capacity)
        data = utils.remove_float64(data)
        lim = 1 - 1e-8
        data["action_1"] = np.clip(data["action_1"], a_min=-lim, a_max=lim)
        data["action_2"] = np.clip(data["action_2"], a_min=-lim, a_max=lim)

        self.data = data
        self.data_size, self.data_segment_length = data["action_1"].shape[:2]

    def __len__(self):
        return self.data_size

    def sample_idx(self, idx):
        idx = np.squeeze(idx)
        is_batch = len(idx.shape) > 0
        if self.segment_length is not None:
            start_idx = np.random.randint(self.data_segment_length - self.segment_length)
            end_idx = start_idx + self.segment_length
        else:
            start_idx, end_idx = 0, self.data_segment_length
        batch = {
            "obs_1": self.data["obs_1"][idx, start_idx:end_idx],   # like (8, 100, 11)
            "obs_2": self.data["obs_2"][idx, start_idx:end_idx],   # like (8, 100, 11)
            "action_1": self.data["action_1"][idx, start_idx:end_idx],   # like (8, 100, 3)
            "action_2": self.data["action_2"][idx, start_idx:end_idx],   # like (8, 100, 3)
            "script_reward_1": self.data["script_reward_1"][idx, start_idx:end_idx][:, :, None],    # like (8, 100, 1)
            "script_reward_2": self.data["script_reward_2"][idx, start_idx:end_idx][:, :, None],    # like (8, 100, 1)
            "label": self.data["label"][idx][:, None],   # like (8, 1)
            "terminal_1": np.zeros([len(idx), end_idx-start_idx, 1], dtype=np.float32) \
                if is_batch else np.zeros([end_idx-start_idx, 1], dtype=np.float32),   # like (8, 100, 1)
            "terminal_2": np.zeros([len(idx), end_idx-start_idx, 1], dtype=np.float32) \
                if is_batch else np.zeros([end_idx-start_idx, 1], dtype=np.float32),   # like (8, 100, 1)
        }
        # post process for soft mode
        if self.mode == "soft":
            r1, r2 = self.reward_scale * batch["script_reward_1"], self.reward_scale * batch["script_reward_2"]
            logits = r2.sum(axis=1) - r1.sum(axis=1)
            prob = 1 / (1 + np.exp(-logits))
            batch["label"] = np.random.binomial(1, prob, size=batch["label"].shape).astype(np.float32)
        return batch

    def __iter__(self):
        while True:
            idxs = np.random.randint(0, len(self), size=self.batch_size)
            yield self.sample_idx(idxs)

    def create_sequential_iter(self):
        start, end = 0, min(self.batch_size, self.data_size)
        while start < self.data_size:
            idxs = list(range(start, min(end, self.data_size)))
            yield self.sample_idx(idxs)
            start += self.batch_size
            end += self.batch_size
