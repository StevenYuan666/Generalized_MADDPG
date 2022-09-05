from tqdm import tqdm
from mamujoco_maddpg.mamujoco_runner import Runner
from mamujoco_maddpg.common.arguments import get_args
from mamujoco_maddpg.common.utils import make_env
import numpy as np
import random
import torch
import matplotlib
import wandb

if __name__ == '__main__':
    # get the params
    seed = [0, 100, 200, 300, 400]
    scenario_names = ["Reacher-v2", "Walker2d-v2"]
    scenario_name = scenario_names[0]
    for load_unload in range(0,2):
        print(" The load status: "+ str(load_unload))
        for run in range(0,5):
            # initial setting
            print(" The run: " + str(run))
            random.seed(seed[run])
            np.random.seed(seed[run])
            torch.manual_seed(seed[run])
            wandb.init(project="Generalized_MADDPG", entity="aims-hign",
                       group="mujoco_eval_rate=1000_if_load_meta:"+str(load_unload), job_type="seed"+str(seed[run])) #id=wandb.util.generate_id(), resume="allow"
            CHECKPOINT_PATH = './checkpoint_load_'+str(load_unload)+'seed'+str(seed[run])+'.tar'
            # args setting
            args = get_args()
            args.scenario_name = scenario_name
            if scenario_name == "Reacher-v2":
                args.agent_conf = "2x1"
                args.agent_obsk = 1
            elif scenario_name == "Walker2d-v2":
                args.agent_conf = "2x3"
                args.agent_obsk = 1

            args.load_meta = load_unload
            env, args = make_env(args)
            runner = Runner(args, env)
            s = runner.env.reset()

            matplotlib.use("Agg")  # avoid "fail to allcoate bitmap"
            val_return = 0
            train_rewards = 0
            returns = []
            episode = 0
            eval_episode = 0
            for time_step in tqdm(range(args.time_steps)):
                u = []
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(runner.agents):
                        action = agent.select_action(s[agent_id], runner.noise, runner.epsilon)
                        u.append(action)
                        actions.append(action)
                for i in range(runner.args.n_agents, runner.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                r, done, info = runner.env.step(actions)
                s_next = runner.env.get_obs()
                runner.buffer.store_episode(s[:runner.args.n_agents], u, r, s_next[:runner.args.n_agents])
                s = s_next
                train_rewards += r[0]
                if runner.buffer.current_size >= runner.args.batch_size:
                    transitions = runner.buffer.sample(runner.args.batch_size)
                    for agent in runner.agents:
                        other_agents = runner.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)  # learn proces
                # evaluate
                if time_step > 0 and time_step % runner.args.evaluate_rate == 0:
                    val_return = runner.evaluate()
                    eval_episode += 1
                    # wandb log val return
                    val_metric = {
                        "val_return": val_return,
                        "eval_episode": eval_episode
                               }
                    wandb.log(val_metric)
                # reset the environment
                if time_step > 0 and time_step % runner.episode_limit == 0:
                    s = runner.env.reset()
                    episode += 1
                    train_metric = {
                        "train_rewards": train_rewards,
                        "episode": episode
                    }
                    wandb.log(train_metric)

                    train_rewards = 0

                runner.noise = max(0.05, runner.noise - 0.0000005)
                runner.epsilon = max(0.05, runner.epsilon - 0.0000005)

            wandb.finish()

