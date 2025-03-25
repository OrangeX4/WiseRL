import itertools
import os
from operator import itemgetter
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn

from wiserl.algorithm.base import Algorithm
import wiserl.module
from wiserl.module.actor import DeterministicActor, GaussianActor


class OracleRM(Algorithm):
    def __init__(
        self,
        *args,
        rm_label: bool = True,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rm_label = rm_label
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def select_reward(self, batch, deterministic=False):
        return torch.concat([batch["script_reward_1"], batch["script_reward_2"]], dim=0).detach()
    
    def select_action(self, batch, deterministic: bool=True):
        raise NotImplementedError

    @torch.no_grad()
    def pretrain_step(self, batches, step: int, total_steps: int) -> Dict:
        batch = batches[0]
        r1, r2 = batch['script_reward_1'].unsqueeze(0), batch['script_reward_2'].unsqueeze(0)
        all_reward = torch.concat([r1, r2], dim=0)
        logits = r2.sum(dim=2) - r1.sum(dim=2)
        labels = batch["label"].float().unsqueeze(0).expand_as(logits)
        reward_loss = self.reward_criterion(logits, labels).sum(0).mean()
        reg_loss = (r1**2).sum(0).mean() + (r2**2).sum(0).mean()
        reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()
        # Calculate win/lose rewards based on labels
        win_mask = labels > 0.5
        win_reward = torch.where(win_mask, r2.squeeze(), r1.squeeze())
        lose_reward = torch.where(win_mask, r1.squeeze(), r2.squeeze())

        metrics = {
            "loss/reward_loss": reward_loss.item(),
            "loss/reward_reg_loss": reg_loss.item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/reward_mean": all_reward.mean().item(),
            "misc/reward_std": all_reward.std().item(),
            "misc/win_reward_mean": win_reward.mean().item(),
            "misc/win_reward_std": win_reward.std().item(),
            "misc/lose_reward_mean": lose_reward.mean().item(),
            "misc/lose_reward_std": lose_reward.std().item(),
        }
        return metrics

    def train_step(self, batches, step: int, total_steps: int) -> Dict:
        return {}