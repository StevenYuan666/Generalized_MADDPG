from mamujoco_maddpg.maml_mamujoco.task import Task
from mamujoco_maddpg.common.utils import make_env
import random
import time
import copy


class TaskSampler:
    def __init__(self, args, batch_size):
        self.args = copy.copy(args)
        # self.scenarios_names = ["simple_spread"]
        self.scenarios_names = ["2_Agent_Ant", "2_Agent_HalfCheetah", "2_Agent_Swimmer", "2_Agent_Humanoid"]
        self.batch_size = batch_size
        self.input_shape = []
        for s in self.scenarios_names:
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

    def sample(self, num_tasks):
        random.seed(int(time.time()))
        tasks = []

        for s in self.scenarios_names:
            args = copy.copy(self.args)
            if s == "2_Agent_Ant":
                scenario_name = "Ant-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "2x4"
                args.agent_obsk = 1
                tasks.append(Task(scenario_name=scenario_name, input_shape=self.input_shape, batch_size=self.batch_size, args=args))
            elif s == "2_Agent_HalfCheetah":
                scenario_name = "HalfCheetah-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "2x3"
                args.agent_obsk = 1
                tasks.append(Task(scenario_name=scenario_name, input_shape=self.input_shape, batch_size=self.batch_size, args=args))
            elif s == "2_Agent_Swimmer":
                scenario_name = "Swimmer-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "2x1"
                args.agent_obsk = 1
                tasks.append(Task(scenario_name=scenario_name, input_shape=self.input_shape, batch_size=self.batch_size, args=args))
            elif s == "2_Agent_Humanoid":
                scenario_name = "Humanoid-v2"
                args.scenario_name = scenario_name
                args.agent_conf = "9|8"
                args.agent_obsk = 1
                tasks.append(Task(scenario_name=scenario_name, input_shape=self.input_shape, batch_size=self.batch_size, args=args))
        # for _ in range(num_tasks):
        #     scenario = random.choice(self.scenarios_names)
        #     args = copy.copy(self.args)
        #     args.scenario_name = scenario
        #     tasks.append(Task(scenario_name=scenario, num_agents=self.num_agents, batch_size=self.batch_size,
        #                       args=args))
        #
        # random.shuffle(tasks)
        return tasks

    # def sample_test(self):
    #     args = copy.copy(self.args)
    #     args.scenario_name = "simple_test"
    #     return Task(scenario_name="simple_test", batch_size=self.batch_size, args=args)
