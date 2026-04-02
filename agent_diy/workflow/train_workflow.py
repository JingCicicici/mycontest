#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time

import numpy as np
import torch

from common_python.utils.common_func import Frame
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import reward_shaping, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    PPO training workflow.
    """
    try:
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]
        monitor_data = {"reward": 0}
        last_report_monitor_time = 0
        last_get_training_metrics_time = 0

        logger.info("Start PPO Training...")
        start_time = time.time()
        last_save_model_time = start_time

        total_reward = 0.0
        episode_count = 0
        win_count = 0

        for episode in range(Config.EPISODES):
            if time.time() - last_get_training_metrics_time > 15:
                last_get_training_metrics_time = time.time()
                training_metrics = get_training_metrics()
                if training_metrics:
                    logger.info(f"training_metrics is {training_metrics}")

            env_obs = env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, logger):
                continue

            obs_data = agent.observation_process(env_obs)
            done = False
            episode_reward = 0.0
            trajectory = []

            while not done:
                act_data = agent.predict(list_obs_data=[obs_data])[0]
                current_action = agent.action_process(act_data)
                next_env_reward, next_env_obs = env.step(current_action)

                if handle_disaster_recovery(next_env_obs, logger):
                    break

                terminated = next_env_obs["terminated"]
                truncated = next_env_obs["truncated"]
                done = terminated or truncated

                reward = reward_shaping(next_env_reward, next_env_obs)
                episode_reward += reward

                trajectory.append(
                    Frame(
                        obs=obs_data.feature,
                        action=current_action,
                        log_prob=act_data.log_prob,
                        reward=reward,
                        done=done,
                        value=act_data.value,
                    )
                )

                if terminated:
                    win_count += 1

                if not done:
                    obs_data = agent.observation_process(next_env_obs)

            if not trajectory:
                continue

            last_value = 0.0
            if not trajectory[-1].done:
                obs_tensor = torch.as_tensor(obs_data.feature, dtype=torch.float32, device=agent.device)
                with torch.no_grad():
                    _, bootstrap_value = agent.model(obs_tensor)
                last_value = float(bootstrap_value.squeeze(0).item())

            processed_samples = sample_process(trajectory)
            returns, advantages = _compute_gae(processed_samples, last_value)
            for sample, ret, adv in zip(processed_samples, returns, advantages):
                sample["return"] = ret
                sample["advantage"] = adv

            learn_metrics = agent.learn(processed_samples)

            total_reward += episode_reward
            episode_count += 1
            now = time.time()
            is_converged = win_count / (episode + 1) > 0.9 and episode > 100

            if is_converged or now - last_report_monitor_time > 15:
                avg_reward = total_reward / max(episode_count, 1)
                logger.info(
                    f"Episode: {episode + 1}, Avg Reward: {avg_reward}, "
                    f"Loss: {learn_metrics.get('loss', 0.0):.4f}, "
                    f"Policy Loss: {learn_metrics.get('policy_loss', 0.0):.4f}, "
                    f"Value Loss: {learn_metrics.get('value_loss', 0.0):.4f}"
                )
                logger.info(f"Training Win Rate: {win_count / (episode + 1)}")
                monitor_data["reward"] = avg_reward
                if monitor:
                    monitor.put_data({os.getpid(): monitor_data})

                total_reward = 0.0
                episode_count = 0
                last_report_monitor_time = now

            if is_converged:
                logger.info(f"Training Converged at Episode: {episode + 1}")
                break

            if now - last_save_model_time > 300:
                logger.info(f"Saving Model at Episode: {episode + 1}")
                agent.save_model()
                last_save_model_time = now

        end_time = time.time()
        logger.info(f"Training Time for {episode + 1} episodes: {end_time - start_time} s")
        agent.save_model()

    except Exception as e:
        raise RuntimeError(f"workflow error: {e}")


def _compute_gae(samples, last_value):
    returns = []
    advantages = []
    gae = 0.0
    next_value = last_value

    for sample in reversed(samples):
        mask = 1.0 - float(sample["done"])
        delta = sample["reward"] + Config.GAMMA * next_value * mask - sample["value"]
        gae = delta + Config.GAMMA * Config.GAE_LAMBDA * mask * gae
        advantages.append(gae)
        returns.append(gae + sample["value"])
        next_value = sample["value"]

    returns.reverse()
    advantages.reverse()
    return np.asarray(returns, dtype=np.float32), np.asarray(advantages, dtype=np.float32)
