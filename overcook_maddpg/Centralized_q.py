import torch
import torch.nn as nn
import torch.nn.functional as F
from overcooked_ai.src.overcooked_ai_py.env import OverCookedEnv
import copy
from overcook_maddpg.model.utils.model import fanin_init

class Centralized_q(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, cfg, sampler=None, hidden_dim=64, activation=F.relu,
                 constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Centralized_q, self).__init__()
        Envs = []
        for scenario in sampler.scenarios_names:
            # cfg_ = copy.copy(cfg)
            cfg.env = scenario
            env = OverCookedEnv(scenario=cfg.env, episode_length=cfg.episode_length)
            Envs.append(env)
        Shape = []
        for e in Envs:
            Shape.append(e.observation_space.n + e.action_space.n)

        input_dim = max(Shape)
        self.input_dim = input_dim * 2
        self.norm1 = nn.BatchNorm1d(self.input_dim)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

        self.activation = activation

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())


    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # X = self.norm1(X)
        h1 = self.activation(self.fc1(X))
        h2 = self.activation(self.fc2(h1))
        h3 = self.activation(self.fc3(h2))
        h4 = self.activation(self.fc4(h3))
        out = self.fc5(h4)
        return out