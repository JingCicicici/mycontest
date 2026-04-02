#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device, logger):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger

    def learn(self, list_sample_data):
        if not list_sample_data:
            return {}

        obs = torch.as_tensor(
            np.asarray([sample["obs"] for sample in list_sample_data], dtype=np.float32),
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray([sample["action"] for sample in list_sample_data], dtype=np.int64),
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            np.asarray([sample["log_prob"] for sample in list_sample_data], dtype=np.float32),
            device=self.device,
        )
        returns = torch.as_tensor(
            np.asarray([sample["return"] for sample in list_sample_data], dtype=np.float32),
            device=self.device,
        )
        advantages = torch.as_tensor(
            np.asarray([sample["advantage"] for sample in list_sample_data], dtype=np.float32),
            device=self.device,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        sample_count = obs.shape[0]
        batch_size = min(Config.BATCH_SIZE, sample_count)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}
        update_steps = 0

        for _ in range(Config.PPO_EPOCHS):
            indices = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                logits, values = self.model(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - Config.CLIP_EPSILON, 1.0 + Config.CLIP_EPSILON
                )
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = clipped_ratio * batch_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)
                loss = (
                    policy_loss
                    + Config.VALUE_COEF * value_loss
                    - Config.ENTROPY_COEF * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), Config.MAX_GRAD_NORM)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["loss"] += float(loss.item())
                update_steps += 1

        if update_steps > 0:
            for key in metrics:
                metrics[key] /= update_steps

        return metrics
