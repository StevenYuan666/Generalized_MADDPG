import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_test", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=40000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--device", type=str, default='cpu', help="device")

    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer_size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./model_maml", help="directory in which training state and model_maml should be saved")
    parser.add_argument("--save_rate", type=int, default=2000, help="save model_maml once every time this many episodes are completed")
    parser.add_argument("--model_dir", type=str, default="", help="directory in which training state and model_maml are loaded")

    # Evaluate
    parser.add_argument("--evaluate_episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate_episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model_maml")
    parser.add_argument("--evaluate_rate", type=int, default=100, help="how often to evaluate model_maml")

    # test
    parser.add_argument("--run-index", type=int, default=0, help="the times of test")
    parser.add_argument("--load-meta", type=int, default=0, help="whether load the meta model_maml")
    args = parser.parse_args()

    return args
