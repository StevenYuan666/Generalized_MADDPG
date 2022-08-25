import torch
from model.network import MLPNetwork, FeedForwardNN
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

        self.centralized_q = FeedForwardNN(204, 1)
        self.centralized_q_optim = torch.optim.Adam(self.centralized_q.parameters(), lr=self.outer_lr)
        self.train_step = 0
        self.total_training_step = 20000
        self.update_times = 2000
        self.episode_limit = 500
        self.num_tasks = 4
        self.save_rate = 10
        self.load_rate = 1

    def train(self):
        result = []
        for i in range(self.total_training_step):
            print("Meta Training " + str(i + 1) + " sampling " + str(self.num_tasks) + " tasks")
            tasks = self.task_sampler.sample()
            for time_step in range(self.update_times):
                total_q_loss = None
                for j, t in enumerate(tasks):
                    for a in t.agent.agents:
                        # if time_step == 0:
                        #     a.target_critic.load_state_dict(self.centralized_q.state_dict())
                        if time_step % self.load_rate == 0:
                            a.critic.load_state_dict(self.centralized_q.state_dict())
                    # inner training
                    task_q_loss = t.run(centralized_q=self.centralized_q)
                    if total_q_loss is None:
                        total_q_loss = task_q_loss
                    else:
                        total_q_loss = total_q_loss.add(task_q_loss)
                if total_q_loss is not None:
                    self.centralized_q_optim.zero_grad()
                    total_q_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.centralized_q.parameters(), 0.5)
                    self.centralized_q_optim.step()
            returns = []
            train_rewards = []
            for t in tasks:
                r = t.evaluate()
                returns.append(r)
                train_rewards.append(t.whole_rewards)

            to_save = [i + 1, np.mean(returns), returns, train_rewards]
            result.append(to_save)

            if i % self.save_rate == 0:
                print("Saving training information and meta centralized q function parameters", end=" ")
                np.save("/Users/stevenyuan/Documents/McGill/CPSL-Lab/Generalized_MARL/Generalized_MADDPG/overcook_maddpg/result/training_info.npy", np.array(result))
                torch.save(self.centralized_q.state_dict(), '/Users/stevenyuan/Documents/McGill/CPSL-Lab/Generalized_MARL/Generalized_MADDPG/overcook_maddpg/result/centralized_q_params.pth')
                print("and successfully saved")

            avg_train_reward = [np.mean(r) for r in train_rewards]
            # wandb.log({"asymmetric_advantages_train_rewards": avg_train_reward[0],
            #            "cramped_room_train_rewards": avg_train_reward[1],
            #            "coordination_ring_train_rewards": avg_train_reward[2],
            #            "counter_circuit_train_rewards": avg_train_reward[3],
            #            "inner_batch_avg_validation_return": np.mean(returns),
            #            "asymmetric_advantages_validation_rewards": returns[0],
            #            "cramped_room_validation_rewards": returns[1],
            #            "coordination_ring_validation_rewards": returns[2],
            #            "counter_circuit_validation_rewards": returns[3]})
            #
            # # Optional
            # wandb.watch(self.centralized_q)

            print("Meta Update: " + str(i + 1), "\n\tTraining_Avg_Rewards: ", avg_train_reward,
                  "\n\tinner_batch_avg_validation_return: " + str(np.mean(returns)),
                  "\n\t" + str([t.cfg.env for t in tasks]) + ": " + str(returns))

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
    # import wandb
    # wandb.init(project="Generalized_MADDPG")
    # wandb.config = {
    #     "meta_q_learning_rate": 0.001,
    #     "total_training_step": 20000,
    #     "update_times_in_task": 2000,
    #     "episode_limit": 500,
    #     "load_rate": 1,
    #     "num_tasks": 4,
    #     "num_hidden_layers": 5,
    #     "size_hidden_layers": 64,
    #     "DDPG": True,
    #     "PPO": False,
    #     "critic_lr": 0.001,
    #     "actor_lr": 0.0001,
    # }
    main()
