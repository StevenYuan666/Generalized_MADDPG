
from task import Task

import random
import time
import copy


class TaskSampler:
    def __init__(self, cfg):
        self.cfg = copy.copy(cfg)

        # self.scenarios_names = ["simple_spread"]
        self.scenarios_names = ["asymmetric_advantages", "cramped_room", "coordination_ring", "counter_circuit"]

    def sample(self):
        tasks = []
        for s in self.scenarios_names:
            cfg = copy.copy(self.cfg)
            cfg.env = s
            tasks.append(Task(cfg))


        return tasks

    def sample_test(self):
        cfg = copy.copy(self.cfg)
        cfg.scenario_name = "forced_coordination"

        return Task(cfg=cfg)

