#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from torch import nn


class Model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        hidden_size = 256

        self.encoder = nn.Sequential(
            nn.Linear(state_shape, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_shape)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        hidden = self.encoder(obs)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value
