from runner import Runner
from common.arguments import get_args
from common.utils import make_env, make_overcook_env
import numpy as np
import random
import torch
import matplotlib


if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_overcook_env(args)
    runner = Runner(args, env)
    seed = [0, 1, 2, 3, 4]
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    matplotlib.use("Agg") # avoid "fail to allcoate bitmap"
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
        
    else:
        runner. run()
