import numpy as np
import inspect
import functools
from multiagent.multi_discrete import MultiDiscrete
from gym import spaces
from mamujoco_maddpg.common.arguments import get_args

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    from multiagent_mujoco.mujoco_multi import MujocoMulti
    env_args = {"scenario": args.scenario_name,
                "agent_conf": args.agent_conf,
                "agent_obsk": args.agent_obsk,
                "episode_limit": args.max_episode_len,
                }
    # print(env_args)
    multi_env = MujocoMulti(env_args=env_args)
    env_info = multi_env.get_env_info()

    args.n_agents = env_info["n_agents"]
    args.obs_shape = [multi_env.get_obs_size() for _ in range(args.n_agents)] # 每一维代表该agent的obs维度
    args.action_shape = [multi_env.action_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的act维度

    args.high_action = max(multi_env.env.action_space.high)
    args.low_action = min(multi_env.env.action_space.low)
    return multi_env, args

if __name__ == "__main__":
    args = get_args()
    env, args = make_env(args)
    print("Hello")