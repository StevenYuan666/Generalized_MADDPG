import numpy as np
import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from model.network import FeedForwardNN
from model.utils.model import *


class Agent(nn.Module):
    '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

    def __init__(self, env_specs=None):
        # self.env_specs = env_specs
        # self.env = gym.make('Hopper-v2')

        super(Agent, self).__init__()
        """
    			Initializes the PPO model, including hyperparameters.
    			Parameters:
    				policy_class - the policy class to use for our actor/critic networks.
    				env - the environment to train on.
    				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
    			Returns:
    				None
    		"""

        # Make sure the environment is compatible with our code
        # assert (type(self.env_specs["observation_space"]) == gym.spaces.Box)
        # assert (type(self.env_specs["action_space"]) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters()

        self.policy_class = FeedForwardNN

        # Extract environment information
        # self.obs_dim = 204
        self.act_dim = 6

        # Initialize actor and critic networks
        self.actor = self.policy_class(96, 6)  # ALG STEP 1
        self.critic = self.policy_class(204, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rews = []
        self.batch_rtgs = []
        self.batch_lens = []
        self.ep_rews = []

        self.log_prob = 0

        # self.t_so_far = 0  # Timesteps simulated so far
        self.i_so_far = 0  # Iterations ran so far
        self.t = 0

        self.reward_record = []
        self.actor_loss_record = []
        self.critic_loss_record = []

    def load_weights(self, root_path):
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
        pass

    def act(self, curr_obs, mode='eval'):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        # print(curr_obs)
        mean = self.actor(curr_obs)
        # print(mean)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM

        if mode == 'eval':
            # use smaller variance for evaluation
            dist = MultivariateNormal(mean, self.cov_mat / 50.0)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()
        print(action)
        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)
        self.log_prob = log_prob
        # Return the sampled action and the log probability of that action in our distribution
        action = onehot_from_logits(action)
        action = onehot_to_number(action)
        action = action.detach().cpu().numpy()
        return action

    def update(self, curr_obs, action, reward, next_obs, done, log_prob):
        # action = number_to_onehot(action)

        self.batch_obs = curr_obs
        self.batch_acts = action
        self.batch_log_probs = log_prob
        self.batch_rews = reward


        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = self.batch_obs.clone().detach()
        batch_acts = self.batch_acts.clone().detach()
        batch_log_probs = self.batch_log_probs.clone().detach()
        batch_rtgs = self.compute_rtgs(self.batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        # self.logger['batch_rews'] = self.batch_rews
        # self.logger['batch_lens'] = self.batch_lens

        self.i_so_far += 1

        # Logging timesteps so far and iterations so far
        # self.logger['t_so_far'] = self.t_so_far
        self.logger['i_so_far'] = self.i_so_far

        # Calculate advantage at k-th iteration
        V, _ = self.evaluate(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()  # ALG STEP 5

        # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
        # isn't theoretically necessary, but in practice it decreases the variance of
        # our advantages and makes convergence much more stable and faster. I added this because
        # solving some environments was too unstable without it.
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        batch_aloss = np.zeros(self.n_updates_per_iteration)
        batch_closs = np.zeros(self.n_updates_per_iteration)

        # This is the loop where we update our network for some n epochs
        for i in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            # NOTE: we just subtract the logs, which is the same as
            # dividing the values and then canceling the log with e^log.
            # For why we use log probabilities instead of actual probabilities,
            # here's a great explanation:
            # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
            # TL;DR makes gradient ascent easier behind the scenes.
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            ratios = ratios.view(2, 256)
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
            # the performance function, but Adam minimizes the loss. So minimizing the negative
            # performance function maximizes it.
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Log actor loss
            self.logger['actor_losses'].append(actor_loss.detach())
            # self.reward_record.append()

            batch_aloss[i] = actor_loss.detach()
            batch_closs[i] = critic_loss.detach()

        # Print a summary of our training so far
        # self._log_summary()

        aloss = np.mean(batch_aloss)
        closs = np.mean(batch_closs)
        # Save our model if it's time
        if self.i_so_far % self.save_freq == 0:
            torch.save(self.actor.state_dict(), './ppo_actor.pth')
            torch.save(self.critic.state_dict(), './ppo_critic.pth')


        self.t = 0
        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rews = []
        self.batch_rtgs = []
        self.batch_lens = []
        self.ep_rews = []

        return aloss, closs

    def compute_rtgs(self, batch_rews):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.
        Parameters:
            batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
        Return:
            batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
    """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.
        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)
        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
    """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        critic_in = torch.cat((batch_obs, batch_acts), dim=2).view(256, -1)
        V = self.critic(critic_in).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self):
        """
        Initialize default and custom values for hyperparameters
        Parameters:
            hyperparameters - the extra arguments included when creating the PPO model, should only include
                                hyperparameters defined below with custom values.
        Return:
            None
    """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 5000  # Number of timesteps to run per batch 最好卡一个能被十万整除的次数
        self.max_timesteps_per_episode = 500  # Max number of timesteps per episode
        self.n_updates_per_iteration = 10  # Number of times to update actor/critic per iteration
        self.lr = 3e-4  # Learning rate of actor optimizer
        self.gamma = 0.99  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.
        Parameters:
            None
        Return:
            None
    """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []