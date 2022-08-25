import torch
import numpy as np

from utils.misc import soft_update

from model.PPOAgent import Agent
from model.utils.model import *
from copy import deepcopy
import torch.nn as nn


class MAPPO(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        self.tau = params.tau

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)

        self.mse_loss = torch.nn.MSELoss()

        # Reshape critic input shape for shared observation
        params.critic.obs_dim = (self.obs_dim + self.action_dim) * self.num_agents

        self.agents = [Agent() for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

    def act(self, observations, sample=False):
        observations = torch.Tensor(observations).to(self.device)

        actions = []
        for agent, obs in zip(self.agents, observations):
            actions.append(agent.act(curr_obs=obs, mode="train").squeeze())
        return np.array(actions)

    def update(self, replay_buffer, logger, step):

        sample = replay_buffer.sample(self.batch_size, nth=self.agent_index)
        obses, actions, rewards, next_obses, dones, log_prob = sample
        to_return = []

        if self.discrete_action:  actions = number_to_onehot(actions)

        for agent_i, agent in enumerate(self.agents):
            agent.update(curr_obs=obses, action=actions, reward=rewards[:, agent_i],
                         next_obs=next_obses, done=dones[:, agent_i], log_prob=log_prob[:, agent_i])

            critic_in = torch.cat((obses, actions), dim=2).view(self.batch_size, -1)
            to_return.append(deepcopy(critic_in))

        return to_return

    def update_all_targets(self):
        for agent in self.agents:
            soft_update(agent.target_critic, agent.critic, self.tau)
            soft_update(agent.target_policy, agent.policy, self.tau)

    def save(self, step):
        # os.mk
        #
        # for i, agent in self.agents:
        #     name = '{0}_{1}_{step}.pth'.format(self.name, i, step)
        #     torch.save(agent, )
        #
        #
        # raise NotImplementedError
        pass

    def load(self, filename):

        raise NotImplementedError

    @property
    def policies(self):
        return [agent.policy for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_policy for agent in self.agents]

    @property
    def critics(self):
        return [agent.critic for agent in self.agents]

    @property
    def target_critics(self):
        return [agent.target_critic for agent in self.agents]