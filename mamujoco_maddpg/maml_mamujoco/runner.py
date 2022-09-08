from mamujoco_maddpg.common.arguments import get_args
from common.utils import make_env
from mamujoco_maddpg.maml_mamujoco.metalearner import MetaLearner
from mamujoco_maddpg.maml_mamujoco.task_sampler import TaskSampler
import numpy as np
import wandb


def main():
    args = get_args()
    sampler = TaskSampler(args=args, batch_size=3)
    leaner = MetaLearner(args=args, sampler=sampler)
    leaner.train()


def test():
    result = []
    for i in range(50):
        args = get_args()
        sampler = TaskSampler(args=args, batch_size=3)
        leaner = MetaLearner(args=args, sampler=sampler)
        r = leaner.test()
        result.append(r)
        print("Run " + str(i) + ": ", r)
    return np.mean(result)


if __name__ == '__main__':

    main()
    # print("Vanilla MADDPG model: ", test(load=False))
    # print("Load pre-trained meta-model:", test())

