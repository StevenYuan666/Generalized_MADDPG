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

    seed=[0,100,200,300,400]
    random.seed(0)
    np.random.seed(seed[args.run_index])
    torch.manual_seed(seed[args.run_index])
    env, args = make_env(args)
    # env, args = make_overcook_env(args)
    runner = Runner(args, env)
   
    np.random.seed(0)
    matplotlib.use("Agg") # avoid "fail to allcoate bitmap"
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
        
    else:
        runner. run()
