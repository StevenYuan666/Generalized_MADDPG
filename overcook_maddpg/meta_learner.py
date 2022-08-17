import torch
from model.network import MLPNetwork
import numpy as np
import hydra
from task_sampler import TaskSampler
from omegaconf import DictConfig


class MetaLearner:

    def __init__(self, sampler, gamma=0.95, outer_lr=0.001, tau=1.0):
        self.task_sampler = sampler
        self.gamma = gamma
        self.outer_lr = outer_lr
        self.tau = tau
        # self.to(device)

        self.centralized_q = MLPNetwork(input_dim=204, out_dim=1)
        self.target_centralized_q = MLPNetwork(input_dim=204, out_dim=1)
        self.centralized_q_optim = torch.optim.Adam(self.centralized_q.parameters(), lr=self.outer_lr)
        self.train_step = 0
        self.total_training_step = 20000
        self.update_times = 20000
        self.episode_limit = 500
        self.num_tasks = 4
        self.save_rate = 10

    def train(self):
        result = []
        for i in range(self.total_training_step):
            print("Meta Training " + str(i + 1) + " sampling " + str(self.num_tasks) + " tasks")
            tasks = self.task_sampler.sample()
            for time_step in range(self.update_times):
                total_q_loss = None
                for j, t in enumerate(tasks):
                    for a in t.agent.agents:
                        if time_step == 0:
                            a.target_critic.load_state_dict(self.centralized_q.state_dict())
                        a.critic.load_state_dict(self.centralized_q.state_dict())
                    # inner training
                    task_q_loss = t.run(time_step=time_step, centralized_q=self.centralized_q)
                    if total_q_loss is None:
                        total_q_loss = task_q_loss
                    else:
                        total_q_loss = total_q_loss.add(task_q_loss)
                if total_q_loss is not None:
                    self.centralized_q_optim.zero_grad()
                    total_q_loss.backward()
                    self.centralized_q_optim.step()
            returns = []
            for t in tasks:
                r = t.evaluate()
                returns.append(r)

            to_save = [i + 1, np.mean(returns), returns]
            result.append(to_save)

            if i % self.save_rate == 0:
                print("Saving training information and meta centralized q function parameters", end=" ")
                np.save("/Users/stevenyuan/Documents/McGill/CPSL-Lab/Generalized_MARL/Generalized_MADDPG/overcook_maddpg/result/training_info.npy", np.array(result))
                torch.save(self.centralized_q.state_dict(), '/Users/stevenyuan/Documents/McGill/CPSL-Lab/Generalized_MARL/Generalized_MADDPG/overcook_maddpg/result/centralized_q_params.pth')
                print("and successfully saved")

            print("Meta Update: " + str(i + 1), "\n\tinner_batch_avg_validation_return: " + str(np.mean(returns))
                  + " [asymmetric_advantages, cramped_room, coordination_ring, counter_circuit]: " + str(returns))

    def test(self):
        self.centralized_q.load_state_dict(torch.load('/Users/stevenyuan/Documents/McGill/CPSL-Lab/Generalized_MARL/Generalized_MADDPG/overcook_maddpg/result/centralized_q_params.pth'))
        t = self.task_sampler.sample_test()
        for time_step in range(self.update_times):
            total_q_loss = None

            for a in t.agents:
                if time_step == 0:
                    a.policy.critic_target_network.load_state_dict(self.centralized_q.state_dict())
                a.policy.critic_network.load_state_dict(self.centralized_q.state_dict())
            # inner training
            task_q_loss = t.run(time_step=time_step, centralized_q=self.centralized_q)
            if total_q_loss is None:
                total_q_loss = task_q_loss
            else:
                total_q_loss = total_q_loss.add(task_q_loss)
            if total_q_loss is not None:
                self.centralized_q_optim.zero_grad()
                total_q_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.centralized_q.parameters(), 0.5)
                self.centralized_q_optim.step()
        r = t.evaluate()
        return r


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig):
    sampler = TaskSampler(cfg=cfg)
    leaner = MetaLearner(sampler=sampler)
    leaner.train()


if __name__ == '__main__':
    main()