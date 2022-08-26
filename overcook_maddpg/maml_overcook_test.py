import copy
import time
import cv2
import wandb

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

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        # self.logger = Logger(self.work_dir,
        #                      save_tb=cfg.log_save_tb,
        #                      log_frequency=cfg.log_frequency,
        #                      agent=cfg.agent.name)

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.discrete_action = cfg.discrete_action_space
        self.save_replay_buffer = cfg.save_replay_buffer
        # self.env = NormalizedEnv(make_env(cfg.env, discrete_action=self.discrete_action))
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
        self.step = 0
        self.val_episode = 0

    def evaluate(self):
        average_episode_reward = 0

        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:
                action = self.agent.act(obs, sample=False)
                obs, rewards, done, info = self.env.step(action)
                rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)

                self.video_recorder.record(self.env)

                episode_reward += sum(rewards)[0]
                episode_step += 1

            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes

        wandb.log({"val_average_reward": average_episode_reward,
                   "val_episode": self.val_episode
                   })
        # self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        # self.logger.dump(self.step)

    def run(self):
        if self.cfg.if_load == 1:
            path = "E:/Project/Generalized_MADDPG/overcook_maddpg/result/centralized_q_params.pth"
            for agent in self.agent.agents:
                agent.critic.load_state_dict(torch.load(path))
                agent.target_critic.load_state_dict(torch.load(path))

        episode, episode_reward, done = 0, 0, True

        while self.step < self.cfg.num_train_steps + 1:
            if done or self.step % self.cfg.eval_frequency == 0:

                # if self.step > 0:
                #     self.logger.log('train/duration', time.time() - start_time, self.step)
                #     start_time = time.time()
                #     self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))
                #
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    # self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    self.val_episode += 1
                    # start_time = time.time()
                #
                # self.logger.log('train/episode_reward', episode_reward, self.step)
                wandb.log({
                    "train_average_reward": episode_reward,
                    "train_episode": episode
                })

                obs = self.env.reset()
                self.ou_percentage = max(0, self.ou_exploration_steps - (self.step - self.num_seed_steps)) / self.ou_exploration_steps
                self.agent.scale_noise(self.ou_final_scale + (self.ou_init_scale - self.ou_final_scale) * self.ou_percentage)
                self.agent.reset_noise()

                episode_reward = 0
                episode_step = 0
                episode += 1

                # self.logger.log('train/episode', episode, self.step)

            if self.step < self.cfg.num_seed_steps:
                action = np.array([self.env.action_space.sample() for _ in self.env_agent_types])
                if self.discrete_action: action = action.reshape(-1, 1)
            else:
                agent_observation = obs[self.agent_indexes]
                agent_actions = self.agent.act(agent_observation, sample=True)
                action = agent_actions

            if self.step >= self.cfg.num_seed_steps and self.step >= self.agent.batch_size:
                agent_observations, agent_action = self.agent.update(self.replay_buffer)

            next_obs, rewards, done, info = self.env.step(copy.deepcopy(action))
            rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)

            if episode_step + 1 == self.env.episode_length:
                done = True

            if self.cfg.render:
                cv2.imshow('Overcooked', self.env.render())
                cv2.waitKey(1)

            episode_reward += sum(rewards)[0]

            if self.discrete_action: action = action.reshape(-1, 1)

            dones = np.array([done for _ in self.env.agents]).reshape(-1, 1)

            self.replay_buffer.add(obs, action, rewards, next_obs, dones)

            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step % 5e4 == 0 and self.save_replay_buffer:
                self.replay_buffer.save(self.work_dir, self.step - 1)


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig) -> None:
    seed = [0, 500, 1000]
    for load_unload in range(0,2):
        for run in range(0, 3):
            wandb.init(project="Generalized_MADDPG", entity="aims-hign",
                       group="overcook_forced_coordination_if_load_meta:" + str(load_unload),
                       job_type="seed" + str(seed[run]))  # id=wandb.util.generate_id(), resume="allow"
            _cfg = copy.deepcopy(cfg)
            _cfg.seed = seed[run]
            _cfg.env = "forced_coordination"
            _cfg.if_load = load_unload
            workspace = Workspace(_cfg)
            workspace.run()
            wandb.finish()


if __name__ == '__main__':
    main()