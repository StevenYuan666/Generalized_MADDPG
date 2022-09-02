import torch
import torch.nn as nn
import torch.nn.functional as F
from mamujoco_maddpg.common.utils import make_env
from multiagent.multi_discrete import MultiDiscrete
import copy


class Centralized_q(nn.Module):
    def __init__(self, args, task_sampler):
        super(Centralized_q, self).__init__()
        self.args = args
        self.device = args.device
        self.max_action = args.high_action if hasattr(args, "high_action") else 1
        self.input_shape = []
        for s in task_sampler.scenarios_names:
            args = copy.copy(self.args)
            if s == "2_Agent_Ant":
                scenario_name = "Ant-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "2x4"
                args.agent_obsk = 1
                _, args = make_env(args=args)
                input_shape = sum(args.obs_shape) + sum(args.action_shape)
                self.input_shape.append(input_shape)
            elif s == "2_Agent_HalfCheetah":
                scenario_name = "HalfCheetah-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "2x3"
                args.agent_obsk = 1
                _, args = make_env(args=args)
                input_shape = sum(args.obs_shape) + sum(args.action_shape)
                self.input_shape.append(input_shape)
            elif s == "2_Agent_Swimmer":
                scenario_name = "Swimmer-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "2x1"
                args.agent_obsk = 1
                _, args = make_env(args=args)
                input_shape = sum(args.obs_shape) + sum(args.action_shape)
                self.input_shape.append(input_shape)
            elif s == "2_Agent_Humanoid":
                scenario_name = "Humanoid-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "9|8"
                args.agent_obsk = 1
                _, args = make_env(args=args)
                input_shape = sum(args.obs_shape) + sum(args.action_shape)
                self.input_shape.append(input_shape)

        self.input_shape = max(self.input_shape)

        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        while True:
            if len(x[0]) == self.input_shape:
                break
            x = torch.cat((x, torch.tensor([[0]]*256).to(self.device)), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
