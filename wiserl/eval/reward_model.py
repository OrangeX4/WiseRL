import collections
import os
from typing import Any, Dict, List, Optional, Sequence

import gym
import numpy as np
import torch

from wiserl.dataset import load_dataset
from wiserl.algorithm.base import Algorithm
from wiserl.utils.functional import order_consistency_rate, spearman_rank_correlation, top_k_overlap_rate


@torch.no_grad()
def eval_reward_model(
    env: gym.Env,
    algorithm: Algorithm,
    eval_dataset_kwargs: Optional[Sequence[str]],
    eval_offline_dataset_kwargs: Optional[Dict[str, Any]]=None,
):
    rm_eval_loss = []
    rm_eval_acc = []
    kwargs = eval_dataset_kwargs.copy()
    eval_dataset = load_dataset(**kwargs)
    for batch in eval_dataset.create_sequential_iter():
        batch = algorithm.format_batch(batch)
        batch["obs"] = torch.concat([batch["obs_1"], batch["obs_2"]], dim=0)
        batch["action"] = torch.concat([batch["action_1"], batch["action_2"]], dim=0)
        reward = algorithm.select_reward(batch)
        r1, r2 = torch.chunk(reward, 2, dim=0)
        logit = r2.sum(dim=1) - r1.sum(dim=1)
        label = batch["label"].float()
        reward_loss = algorithm.reward_criterion(logit, label)
        reward_acc = ((logit > 0) == torch.round(label)).float()
        rm_eval_loss.extend(reward_loss)
        rm_eval_acc.extend(reward_acc)
    eval_dataset_results = {
        "val_loss": torch.as_tensor(rm_eval_loss).mean().item(),
        "val_acc": torch.as_tensor(rm_eval_acc).mean().item(),
    }

    eval_offline_dataset_results = {}
    if eval_offline_dataset_kwargs:
        eval_offline_dataset = load_dataset(**eval_offline_dataset_kwargs)
        pred_reward_list = []
        gt_reward_list = []
        for batch in eval_offline_dataset.create_sequential_iter():
            batch = algorithm.format_batch(batch)
            reward = algorithm.select_reward(batch)
            pred_reward_list.append(reward)
            if "mask" in batch:
                reward = reward * batch["mask"].float()
            gt_reward_list.append(batch["reward"])
        pred_return = torch.concat(pred_reward_list, dim=0).sum(dim=1)
        gt_return = torch.concat(gt_reward_list, dim=0).sum(dim=1)
        pred_return = pred_return.reshape(-1).tolist()
        gt_return = gt_return.reshape(-1).tolist()
        eval_offline_dataset_results = {
            "order_consistency_rate": order_consistency_rate(pred_return, gt_return),
            "spearman_rank_correlation": spearman_rank_correlation(pred_return, gt_return),
            "top1%_overlap_rate": top_k_overlap_rate(pred_return, gt_return, percent=0.01),
            "top5%_overlap_rate": top_k_overlap_rate(pred_return, gt_return, percent=0.05),
            "top10%_overlap_rate": top_k_overlap_rate(pred_return, gt_return, percent=0.1),
            "top20%_overlap_rate": top_k_overlap_rate(pred_return, gt_return, percent=0.2),
            "top50%_overlap_rate": top_k_overlap_rate(pred_return, gt_return, percent=0.5),
        }
    return {
        **eval_dataset_results,
        **eval_offline_dataset_results,
    }
