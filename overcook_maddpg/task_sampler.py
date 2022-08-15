from overcook_maddpg.task import Task
import random
import time
import copy


class TaskSampler:
    def __init__(self, cfg):
        self.cfg = copy.copy(cfg)
        self.scenarios_names = ["asymmetric_advantages", "cramped_room", "coordination_ring", "counter_circuit"]

    def sample(self):
        random.seed(int(time.time()))
        tasks = []

        for s in self.scenarios_names:
            cfg = copy.copy(self.cfg)
            cfg.env = s
            t = Task(cfg=cfg)
            tasks.append(t)
        return tasks

    def sample_test(self):
        cfg = copy.copy(self.cfg)
        cfg.scenario_name = "forced_coordination"
        return Task(cfg=cfg)
