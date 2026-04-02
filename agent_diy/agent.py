#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
import torch
from torch.distributions import Categorical
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.model.model import Model

ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", act=None, log_prob=None, value=None)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger
        self.action_size = Config.ACTION_SIZE
        self.obs_shape = Config.OBSERVATION_SHAPE
        self.device = torch.device(device if device else "cpu")

        self.model = Model(self.obs_shape, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger)

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        self.model.train()
        return [self._policy_action(obs_data.feature, obs_data.legal_action, deterministic=False) for obs_data in list_obs_data]

    def exploit(self, list_obs_data):
        self.model.eval()
        if isinstance(list_obs_data, dict):
            obs_data = self.observation_process(list_obs_data)
        elif isinstance(list_obs_data, list):
            obs_data = list_obs_data[0]
        else:
            obs_data = list_obs_data

        act_data = self._policy_action(
            obs_data.feature,
            obs_data.legal_action,
            deterministic=Config.EVAL_DETERMINISTIC,
        )
        return self.action_process(act_data)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, raw_obs):
        obs = raw_obs["observation"]
        frame_state = obs["frame_state"]
        hero_pos = frame_state["hero"]["pos"]
        pos_x = float(hero_pos["x"])
        pos_z = float(hero_pos["z"])

        end_pos = obs["game_info"]["end"]
        end_x = float(end_pos["x"])
        end_z = float(end_pos["z"])

        feature = [
            pos_x / 63.0,
            pos_z / 63.0,
            end_x / 63.0,
            end_z / 63.0,
            (end_x - pos_x) / 63.0,
            (end_z - pos_z) / 63.0,
            np.sqrt((end_x - pos_x) ** 2 + (end_z - pos_z) ** 2) / 89.1,
        ]

        treasure_status = [0.0] * 10
        treasure_pos = [0.0] * 20
        for organ in frame_state["organs"]:
            if organ["sub_type"] != 1:
                continue
            idx = int(organ["config_id"])
            if 0 <= idx < 10:
                treasure_status[idx] = float(organ["status"])
                treasure_pos[2 * idx] = float(organ["pos"]["x"]) / 63.0
                treasure_pos[2 * idx + 1] = float(organ["pos"]["z"]) / 63.0

        feature.extend(treasure_status)
        feature.extend(treasure_pos)

        grid = frame_state.get("map", {}).get("grid")
        local_map = [0.0] * 121
        if grid:
            cursor = 0
            for dx in range(-5, 6):
                for dz in range(-5, 6):
                    x = int(pos_x) + dx
                    z = int(pos_z) + dz
                    if 0 <= x < len(grid) and 0 <= z < len(grid[x]):
                        local_map[cursor] = float(grid[x][z])
                    cursor += 1
        feature.extend(local_map)

        legal_action = obs.get("legal_action", [1] * self.action_size)
        if len(legal_action) < self.action_size:
            legal_action = list(legal_action) + [1] * (self.action_size - len(legal_action))
        legal_action = [float(v) for v in legal_action[: self.action_size]]
        feature.extend(legal_action)

        env_info = obs.get("env_info", {})
        max_step = max(float(env_info.get("max_step", 2000)), 1.0)
        step_no = float(env_info.get("step_no", 0))
        score = float(env_info.get("score", 0))
        feature.extend([step_no / max_step, score / 100.0])

        while len(feature) < self.obs_shape:
            feature.append(0.0)

        return ObsData(
            feature=np.asarray(feature[: self.obs_shape], dtype=np.float32),
            legal_action=np.asarray(legal_action, dtype=np.float32),
        )

    def action_process(self, act_data):
        return act_data.act

    def save_model(self, path=None, id="1"):
        save_dir = path or "ckpt"
        model_file_path = f"{save_dir}/model.ckpt-{str(id)}.pth"
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        load_dir = path or "ckpt"
        model_file_path = f"{load_dir}/model.ckpt-{str(id)}.pth"
        try:
            checkpoint = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state_dict = checkpoint.get("optimizer_state_dict")
            if optimizer_state_dict:
                self.optimizer.load_state_dict(optimizer_state_dict)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)

    def _policy_action(self, feature, legal_action=None, deterministic=False):
        obs_tensor = torch.as_tensor(feature, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            logits = logits.squeeze(0)
            if legal_action is not None:
                mask = torch.as_tensor(legal_action, dtype=torch.float32, device=self.device)
                if torch.any(mask > 0):
                    logits = logits.masked_fill(mask <= 0, -1e9)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

        return ActData(
            act=int(action.item()),
            log_prob=float(log_prob.item()),
            value=float(value.squeeze(0).item()),
        )
