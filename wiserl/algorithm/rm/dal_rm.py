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


class DAL_RM(Algorithm):
    def __init__(
        self,
        *args,
        reward_reg: float = 0.0,
        rm_label: bool = True,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward_reg = reward_reg
        self.rm_label = rm_label
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_kwargs):
        network = {}
        super().setup_network(network_kwargs)
        reward_act = {
            "identity": nn.Identity(),
            "sigmoid": nn.Sigmoid(),
        }.get(network_kwargs["reward"].pop("reward_act"))
        reward = vars(wiserl.module)[network_kwargs["reward"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["reward"]
        )
        network["reward"] = nn.Sequential(reward, reward_act)
        self.network = nn.ModuleDict(network).to(self.device)


    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        default_kwargs = optim_kwargs.get("default", {})
        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](self.network.reward.parameters(), **reward_kwargs)

    def select_reward(self, batch, deterministic=False):
        obs, action = batch["obs"], batch["action"]
        reward = self.network.reward(torch.concat([obs, action], dim=-1))
        return reward.mean(0).detach()
    
    def select_action(self, batch, deterministic: bool=True):
        raise NotImplementedError

    def pretrain_step(self, batches, step: int, total_steps: int) -> Dict:
        batch = batches[0]
        F_B, F_S = batch["obs_1"].shape[0:2]
        
        # Combine observations and actions from both segments
        all_obs = torch.concat([
            batch["obs_1"].reshape(-1, self.obs_dim),
            batch["obs_2"].reshape(-1, self.obs_dim)
        ])
        all_action = torch.concat([
            batch["action_1"].reshape(-1, self.action_dim),
            batch["action_2"].reshape(-1, self.action_dim)
        ])
        
        labels = batch["label"].float()
        
        self.network.reward.train()
        all_reward = self.network.reward(torch.concat([all_obs, all_action], dim=-1))
        
        reward = all_reward.reshape(-1, 2*F_B, F_S, 1)
        
        # Create combined labels (original labels and their complements) and expand to match reward shape
        all_labels = torch.concat([1-labels, labels], dim=0).unsqueeze(0).unsqueeze(2).expand(-1, -1, F_S, 1)
        reward_loss = self.reward_criterion(reward, all_labels).sum(0).mean()
        
        # Regularization loss
        reg_loss = (all_reward**2).mean()
        
        with torch.no_grad():
            # For accuracy calculation, we still need to sum over sequence dimension
            reward_sum = reward.sum(dim=2)
            all_labels_sum = all_labels[:,:,0,:]  # Take one instance as all are the same after expansion
            win_mask = (all_labels_sum > 0.5)
            lose_mask = (all_labels_sum <= 0.5)
            reward_accuracy = ((reward_sum > 0) == torch.round(all_labels_sum)).float().mean()
            win_reward_accuracy = ((reward_sum > 0) == torch.round(all_labels_sum))[win_mask].float().mean()
            lose_reward_accuracy = ((reward_sum > 0) == torch.round(all_labels_sum))[lose_mask].float().mean()
            
            # Expand masks to match reward shape for reward statistics
            win_mask_expanded = win_mask.unsqueeze(2).expand(-1, -1, F_S, -1)
            lose_mask_expanded = lose_mask.unsqueeze(2).expand(-1, -1, F_S, -1)
            win_reward = reward * win_mask_expanded
            lose_reward = reward * lose_mask_expanded

        self.optim["reward"].zero_grad()
        (reward_loss + self.reward_reg * reg_loss).backward()
        self.optim["reward"].step()

        metrics = {
            "loss/reward_loss": reward_loss.item(),
            "loss/reward_reg_loss": reg_loss.item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/win_reward_accuracy": win_reward_accuracy.item(),
            "misc/lose_reward_accuracy": lose_reward_accuracy.item(),
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

    def load_pretrain(self, path):
        for attr in ["reward"]:
            state_dict = torch.load(os.path.join(path, attr+".pt"), map_location=self.device)
            self.network.__getattr__(attr).load_state_dict(state_dict)

    def save_pretrain(self, path):
        os.makedirs(path, exist_ok=True)
        for attr in ["reward"]:
            state_dict = self.network.__getattr__(attr).state_dict()
            torch.save(state_dict, os.path.join(path, attr+".pt"))
