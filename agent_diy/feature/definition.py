#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


def sample_process(list_game_data):
    return [
        {
            "obs": frame.obs,
            "action": frame.action,
            "log_prob": frame.log_prob,
            "reward": frame.reward,
            "done": frame.done,
            "value": frame.value,
        }
        for frame in list_game_data
    ]


def reward_shaping(env_reward, env_obs):
    score = env_obs["observation"]["env_info"]["score"]
    terminated = env_obs["terminated"]
    truncated = env_obs["truncated"]

    reward = float(env_reward) * 0.2 - 0.01

    if score > 0:
        reward += float(score)

    if terminated:
        reward += 150.0
    elif truncated:
        reward -= 15.0

    return reward
