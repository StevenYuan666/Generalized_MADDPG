from maml_rl.task import Task
import random
import time
import copy


class TaskSampler:
    def __init__(self, args, num_agents, batch_size):
        self.args = copy.copy(args)
        # self.scenarios_names = ["simple_spread"]
        self.scenarios_names = ["simple_reference", "simple_speaker_listener", "simple_spread"]
        self.num_agents = num_agents
        self.batch_size = batch_size

    def sample(self, num_tasks):
        random.seed(int(time.time()))
        tasks = []

        for s in self.scenarios_names:
            args = copy.copy(self.args)
            args.scenario_name = s
            tasks.append(Task(scenario_name=s, num_agents=self.num_agents, input_shape=input_shape, batch_size=self.batch_size, args=args))
        # for _ in range(num_tasks):
        #     scenario = random.choice(self.scenarios_names)
        #     args = copy.copy(self.args)
        #     args.scenario_name = scenario
        #     tasks.append(Task(scenario_name=scenario, num_agents=self.num_agents, batch_size=self.batch_size,
        #                       args=args))
        #
        # random.shuffle(tasks)
        return tasks

    def sample_test(self):
        args = copy.copy(self.args)
        args.scenario_name = "simple_test"
        return Task(scenario_name="simple_test", num_agents=self.num_agents, batch_size=self.batch_size, args=args)
