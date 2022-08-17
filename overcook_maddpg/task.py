import time
import cv2
import torch

from recoder import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
from utils.train import set_seed_everywhere
from utils.environment import get_agent_types

from overcooked_ai.src.overcooked_ai_py.env import OverCookedEnv

from model.utils.model import *

from utils.agent import find_index

import hydra
from omegaconf import DictConfig
import os
import numpy as np


class Task(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        # print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.discrete_action = cfg.discrete_action_space
        self.save_replay_buffer = cfg.save_replay_buffer
        # self.env = NormalizedEnv(make_env(cfg.env, discrete_action=self.discrete_action))
        # print(self.cfg.env)
        self.env = OverCookedEnv(scenario=self.cfg.env, episode_length=self.cfg.episode_length)

        self.env_agent_types = get_agent_types(self.env)
        self.agent_indexes = find_index(self.env_agent_types, 'ally')
        self.adversary_indexes = find_index(self.env_agent_types, 'adversary')

        # OU Noise settings
        self.num_seed_steps = cfg.num_seed_steps
        self.ou_exploration_steps = cfg.ou_exploration_steps
        self.ou_init_scale = cfg.ou_init_scale
        self.ou_final_scale = cfg.ou_final_scale

        if self.discrete_action:
            cfg.agent.params.obs_dim = self.env.observation_space.n
            cfg.agent.params.action_dim = self.env.action_space.n
            cfg.agent.params.action_range = list(range(cfg.agent.params.action_dim))
        else:
            # Don't use!
            cfg.agent.params.obs_dim = self.env.observation_space[0].shape[0]
            cfg.agent.params.action_dim = self.env.action_space[0].shape[0]
            cfg.agent.params.action_range = [-1, 1]

        cfg.agent.params.agent_index = self.agent_indexes
        cfg.agent.params.critic.input_dim = cfg.agent.params.obs_dim + cfg.agent.params.action_dim

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.common_reward = cfg.common_reward
        obs_shape = [len(self.env_agent_types), cfg.agent.params.obs_dim]
        action_shape = [len(self.env_agent_types), cfg.agent.params.action_dim if not self.discrete_action else 1]
        reward_shape = [len(self.env_agent_types), 1]
        dones_shape = [len(self.env_agent_types), 1]
        self.replay_buffer = ReplayBuffer(obs_shape=obs_shape,
                                          action_shape=action_shape,
                                          reward_shape=reward_shape,
                                          dones_shape=dones_shape,
                                          capacity=int(cfg.replay_buffer_capacity),
                                          device=self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        time_step = 0
        self.obs = None
        self.mse_loss = torch.nn.MSELoss()
        self.done = True

    def evaluate(self):
        average_episode_reward = 0

        # self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:
                action = self.agent.act(obs, sample=False)
                obs, rewards, done, info = self.env.step(action)
                rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)
                episode_reward += sum(rewards)[0]
                episode_step += 1

            average_episode_reward += episode_reward

        average_episode_reward /= self.cfg.num_eval_episodes
        return average_episode_reward
        # self.video_recorder.save(f'{time_step}.mp4')
        #
        # self.logger.log('eval/episode_reward', average_episode_reward, time_step)
        # self.logger.dump(time_step)

    def run(self, time_step, centralized_q):

        if (time_step % self.cfg.eval_frequency) == 0 or self.done:
            self.obs = self.env.reset()

            self.ou_percentage = max(0, self.ou_exploration_steps - (
                        time_step - self.num_seed_steps)) / self.ou_exploration_steps
            self.agent.scale_noise(
                self.ou_final_scale + (self.ou_init_scale - self.ou_final_scale) * self.ou_percentage)
            self.agent.reset_noise()

        if time_step < self.cfg.num_seed_steps:
            action = np.array([self.env.action_space.sample() for _ in self.env_agent_types])
            if self.discrete_action: action = action.reshape(-1, 1)
        else:
            agent_observation = self.obs[self.agent_indexes]
            agent_actions = self.agent.act(agent_observation, sample=True)
            action = agent_actions

        if (time_step >= self.cfg.num_seed_steps) and (time_step >= self.agent.batch_size):
            self.agent.update(self.replay_buffer, self.logger, time_step)

        next_obs, rewards, self.done, info = self.env.step(action)

        task_q_loss = None
        if len(self.replay_buffer) > self.agent.batch_size:
            sample = self.replay_buffer.sample(batch_size=self.agent.batch_size, nth=self.agent.agent_index)
            obses, actions, rewards, next_obses, dones = sample
            if self.discrete_action:
                actions = number_to_onehot(actions)
            critic_in = torch.cat((obses, actions), dim=2).view(self.agent.batch_size, -1)
            for a in self.agent.agents:
                target_q = a.critic(critic_in.float())
                q_value = centralized_q(critic_in.float())
                q_loss = self.mse_loss(q_value, target_q)
                if task_q_loss is None:
                    task_q_loss = q_loss
                else:
                    task_q_loss = task_q_loss.add(q_loss)

        rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)
        # if rewards[0][0] > 0: print(rewards[0])

        if (time_step + 1) % self.env.episode_length == 0:
            self.done = True

        if self.discrete_action: action = action.reshape(-1, 1)

        dones = np.array([self.done for _ in self.env.agents]).reshape(-1, 1)

        self.replay_buffer.add(self.obs, action, rewards, next_obs, dones)

        self.obs = next_obs

        return task_q_loss
