import copy
import os
from abc import ABC
from typing import Any, Callable, Dict, Optional, Type, Union

import random
import numpy as np
import torch
from tqdm import tqdm

from common.cns_visualization import traj_visualization_2d
from stable_baselines3.common.dual_variable import DualVariable
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecNormalizeWithCost


class PolicyIterationLagrange(ABC):

    def __init__(self,
                 env: Union[GymEnv, str],
                 max_iter: int,
                 n_actions: int,
                 reward_states: list,
                 height: int,  # table length
                 width: int,  # table width
                 start_states: list,
                 terminal_states: list,
                 stopping_threshold: float,
                 seed: int,
                 gamma: float = 0.99,
                 v0: float = 0.0,
                 budget: float = 0.,
                 apply_lag: bool = True,
                 penalty_initial_value: float = 1,
                 penalty_learning_rate: float = 0.01,
                 penalty_min_value: Optional[float] = None,
                 penalty_max_value: Optional[float] = None,
                 log_file=None
                 ):
        super(PolicyIterationLagrange, self).__init__()
        self.stopping_threshold = stopping_threshold
        self.gamma = gamma
        self.env = env
        self.log_file = log_file
        self.max_iter = max_iter
        self.n_actions = n_actions
        self.start_states = start_states
        self.terminal_states = terminal_states
        self.v0 = v0
        self.seed = seed
        self.height = height
        self.width = width
        self.penalty_initial_value = penalty_initial_value
        self.penalty_min_value = penalty_min_value
        self.penalty_max_value = penalty_max_value
        self.reward_states = reward_states
        self.penalty_learning_rate = penalty_learning_rate
        self.apply_lag = apply_lag
        self.budget = budget
        self.num_timesteps = 0
        
        if self.n_actions == 9:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            #self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu', 8: 's'}
            #self.action_space = gym.spaces.Discrete(9)
        elif self.n_actions == 8:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
            #self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu'}
            #self.action_space = gym.spaces.Discrete(8)
        elif self.n_actions == 4:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3]
            #self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u'}
            #self.action_space = gym.spaces.Discrete(4)
        else:
            raise EnvironmentError("Unknown number of actions {0}.".format(n_actions))
        self._setup_model()
        self.indicator = 5
        self.reward_mat = np.zeros((self.height, self.width))
        for reward_pos in self.reward_states:
            self.reward_mat[reward_pos[0], reward_pos[1]] = 1
        if len(self.start_states)==1:
            self.admissible_actions = self.get_actions(self.start_states[0])
        else:
            self.admissible_actions = None

        #self.env_for_us = None
        

    def _setup_model(self) -> None:
        self.dual = DualVariable(self.budget,
                                 self.penalty_learning_rate,
                                 self.penalty_initial_value,
                                 min_clamp=self.penalty_min_value,
                                 max_clamp=self.penalty_max_value)
        self.v_m = self.get_init_v()
        self.pi = self.get_equiprobable_policy()

    def get_init_v(self):
        v_m = self.v0 * np.ones((self.height, self.width))
        # # Value function of terminal state must be 0
        # v0[self.e_x, self.e_y] = 0
        return v_m

    # eliminate invalid action
    def get_equiprobable_policy(self):
        pi = 1 / self.n_actions * np.ones((self.height, self.width, self.n_actions))
        for x in range(self.height):
            for y in range(self.width):
                if (x==0 or x==6 or y==0 or y==6) and ([x, y] not in self.terminal_states):
                    for action in range(self.n_actions):
                        next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
                        if not ((0<=next_state[0]<self.height) and (0<=next_state[1]<self.width)):
                            pi[x][y][action] = 0
                    pi[x][y] = pi[x][y] * 1/np.sum(pi[x][y])

        #print('pi',pi) 
        #input('pi')
        return pi

    def get_actions(self, state):
        """
        Returns list of actions that can be taken from the given state.
        """
        if self.reward_mat[state[0]][state[1]] in \
                [-np.inf, float('inf'), np.nan, float('nan')]:
            return [4]
        actions = []
        for i in range(len(self.actions)):
            inc = self.neighbors[i]
            a = self.actions[i]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if 0 <= nei_s[0] < self.height and 0 <= nei_s[1] < self.width and \
                    self.reward_mat[nei_s[0]][nei_s[1]] not in \
                    [-np.inf, float('inf'), np.nan, float('nan')]:
                actions.append(a)
        return actions
  
    # get current policy
    def get_policy(self):
        return self.pi

    # get current v_m
    def get_v_m(self):
        return self.v_m

    def learn(self,
              #env_for_us,
              total_timesteps: int,
              cost_function: Union[str, Callable],
              transition: np.ndarray,
              latent_info_str: Union[str, Callable] = '',
              callback=None, ):
        #self.env_for_us = env_for_us
        policy_stable, dual_stable = False, False
        iter = 0
        for iter in tqdm(range(total_timesteps)):
            print('\npolicy_stable',policy_stable,dual_stable,'\n')
            #input('1')
            if policy_stable and dual_stable:
                print("\nStable at Iteration {0}.".format(iter), file=self.log_file)
                break
            self.num_timesteps += 1
            # Run the policy evaluation
            self.policy_evaluation(cost_function, transition)
            #print('self.v_m1',np.round(self.v_m,2))
            #input('self.v_m1')
            # Run the policy improvement algorithm
            policy_stable = self.policy_improvement(cost_function, transition)
            cumu_reward, length, dual_stable, traj, traj_acs = self.dual_update(cost_function)
        logger.record("train/iterations", iter)
        logger.record("train/cumulative rewards", cumu_reward)
        logger.record("train/length", length)
        return self.v_m, traj, traj_acs

    # expert learn its value function, Q-value function, thus advantage function
    def expert_learn(self,
              #env_for_us,
              total_timesteps: int,
              cost_function: Union[str, Callable],
              expert_policy: np.ndarray,
              #v_m: np.ndarray, 
              unsafe_states: list,
              latent_info_str: Union[str, Callable] = '',              
              transition=None,
              callback=None,):

        iter = 0
        self.pi=expert_policy
        #print('self.pi',np.round(self.pi,2))
        #input('self.pi')
        #print('transition',np.round(transition,2))
        #input('transition')
        for iter in tqdm(range(2)):#need only once policy evaluation 
            # Run the policy evaluation
            self.policy_evaluation_for_expert(cost_function, transition, unsafe_states)           
        #print('v_m,self.v_m', '\n', np.round(v_m,2), '\n', np.round(self.v_m,2))
        #input('v_m,self.v_m')
        #self.v_m=v_m
        Q = np.zeros((self.height, self.width, self.n_actions))
        A = np.zeros((self.height, self.width, self.n_actions))
        if transition is not None:
            for i in range(self.height):
                for j in range(self.width):
                    for k in range(self.n_actions):
                        for i1 in range(self.height):
                            for j1 in range(self.width):
                                if transition[i][j][k][i1][j1]!=0:
                                    Q[i][j][k] += self.gamma*transition[i][j][k][i1][j1]*self.v_m[i1][j1]
                        if Q[i][j][k] - self.v_m[i][j] > 0:
                            A[i][j][k] = Q[i][j][k] - self.v_m[i][j] 
        
                    
        return self.v_m, Q, A

    def step(self, action):
        #input('\nstep0')
        return self.env.step(np.asarray([action]))

    def dual_update(self, cost_function):
        """policy rollout for recording training performance"""
        obs = self.env.reset()
        cumu_reward, length = 0, 0
        actions_game, obs_game, costs_game = [], [], []
        while True:
            actions, _ = self.predict(obs=obs, state=None)
            #print('actions',actions)
            actions_game.append(actions[0])
            obs_primes, rewards, dones, infos = self.step(actions)
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                #print('costs',costs)
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                else:
                    orig_costs = costs
                    #print('orig_costs2',orig_costs)
            else:
                costs = cost_function(obs, actions)
                orig_costs = costs #costs # (np.exp(costs)-1)
            self.admissible_actions = infos[0]['admissible_actions']
            costs_game.append(orig_costs)
            obs = obs_primes
            obs_game.append(obs[0])
            done = dones[0]
            if done:
                break
            cumu_reward += rewards[0]
            length += 1
        costs_game_mean = np.asarray(costs_game).mean()
        self.dual.update_parameter(torch.tensor(costs_game_mean))
        penalty = self.dual.nu().item()
        print("Performance: dual {0}, cost: {1}, states: {2}, "
              "actions: {3}, rewards: {4}.".format(penalty,
                                                   costs_game_mean.tolist(),
                                                   np.asarray(obs_game).tolist(),
                                                   np.asarray(actions_game).tolist(),
                                                   cumu_reward),
              file=self.log_file,
              flush=True)
        dual_stable = True if costs_game_mean == 0 and cumu_reward ==1 else False
        return cumu_reward, length, dual_stable,np.asarray(obs_game).tolist(), np.asarray(actions_game).tolist()

    def policy_evaluation(self, cost_function, transition):
        iter = 0

        delta = self.stopping_threshold + 1
        while delta >= self.stopping_threshold and iter <= self.max_iter-1:
            old_v = self.v_m.copy()
            delta = 0

            # Traverse all states
            for x in range(self.height):
                for y in range(self.width):
                    # Run one iteration of the Bellman update rule for the value function
                    self.bellman_update(old_v, x, y, cost_function, transition)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - self.v_m[x, y]))
            iter += 1
        print("\n\nThe Policy Evaluation algorithm converged after {} iterations".format(iter),
              file=self.log_file)

    def policy_evaluation_for_expert(self, cost_function, transition, unsafe_states):
        iter = 0

        delta = self.stopping_threshold + 1
        while delta >= self.stopping_threshold/100 and iter <= 2*self.max_iter-1:
            old_v = self.v_m.copy()
            delta = 0

            # Traverse all states
            for x in range(self.height):
                for y in range(self.width):
                    # Run one iteration of the Bellman update rule for the value function
                    self.bellman_update_for_expert(old_v, x, y, cost_function, transition, unsafe_states)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - self.v_m[x, y]))
            iter += 1
            #print('delta',delta,self.stopping_threshold/10)
            #input('delta')
        print("\n\nThe Policy Evaluation algorithm converged after {} iterations".format(iter),
              file=self.log_file)


    def policy_improvement(self, cost_function, transition):
        """Applies the Policy Improvement step."""
        policy_stable = True
        # Iterate states
        for x in range(self.height):
            for y in range(self.width):
                if [x, y] in self.terminal_states:
                    continue
                old_pi = self.pi[x, y, :].copy()

                # Iterate all actions
                action_values = []
                for action in range(self.n_actions):
                    states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert states[0][0] == x and states[0][1] == y
                    # Compute next state, but not the same
                    # if the action is invalid, continue
                    next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
                    if not ((0<=next_state[0]<self.height) and (0<=next_state[1]<self.width)):
                        action_values.append(-np.inf)
                        continue

                    x_coordinate = np.nonzero(transition[x, y, action])[0]
                    y_coordinate = np.nonzero(transition[x, y, action])[1]
                    curr_val1 = 0
                    for i in range(len(x_coordinate)):
                        s_primes = [[x_coordinate[i], y_coordinate[i]]]
                        rewards = [self.reward_mat[s_primes[0][0]][s_primes[0][1]]]
                        costs = cost_function(np.array(s_primes), [action])
                        orig_costs = costs #costs # (np.exp(costs)-1)
                        current_penalty = self.dual.nu().item()
                        lag_costs = self.apply_lag * current_penalty * orig_costs[0]
                        # Get value
                        curr_val = rewards[0] - lag_costs + self.gamma * self.v_m[s_primes[0][0], s_primes[0][1]]
                        curr_val1 += transition[x, y, action, s_primes[0][0], s_primes[0][1]] * curr_val
                        #print('ii, transition', i, transition[x, y, action, s_primes[0][0], s_primes[0][1]])
                        #input('ii')

                    #curr_val = 1 * curr_val
                    action_values.append(curr_val1)
                #print('action_values',x,y,action_values)
                #input('')
                best_actions = np.argwhere(action_values == np.amax(action_values)).flatten().tolist()
                # Define new policy
                self.define_new_policy(x, y, best_actions)

                # Check whether the policy has changed
                if not (old_pi == self.pi[x, y, :]).all():
                    policy_stable = False
                #if np.sum(abs(old_pi - self.pi[x, y, :]))>0.01:
                    #policy_stable = False

        return policy_stable

    def define_new_policy(self, x, y, best_actions):
        """Defines a new policy given the new best actions.
        Args:
            pi (array): numpy array representing the policy
            x (int): x value position of the current state
            y (int): y value position of the current state
            best_actions (list): list with best actions
            actions (list): list of every possible action
        """

        prob = 1 / len(best_actions)

        for a in range(self.n_actions):
            self.pi[x, y, a] = prob if a in best_actions else 0


    def bellman_update(self, old_v, x, y, cost_function, transition):
        if [x, y] in self.terminal_states:
            return
        total = 0
        for action in range(self.n_actions):
            states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
            assert states[0][0] == x and states[0][1] == y
            # allow only valid action
            next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
            if not ((0<=next_state[0]<self.height) and (0<=next_state[1]<self.width)):
                continue

            x_coordinate = np.nonzero(transition[x, y, action])[0]
            y_coordinate = np.nonzero(transition[x, y, action])[1]
            for i in range(len(x_coordinate)):
                s_primes = [[x_coordinate[i], y_coordinate[i]]]
                rewards = [self.reward_mat[s_primes[0][0]][s_primes[0][1]]]
                costs = cost_function(np.array(s_primes), [action])
                orig_costs = costs # costs # (np.exp(costs)-1)
                gamma_values = self.gamma * old_v[s_primes[0][0], s_primes[0][1]]
                current_penalty = self.dual.nu().item()
                lag_costs = self.apply_lag * current_penalty * orig_costs[0]
                total += self.pi[x, y, action] * transition[x, y, action, s_primes[0][0], s_primes[0][1]]*(rewards[0] - lag_costs + gamma_values)
                #print('i, transition', x, y, action, s_primes, i, rewards[0], self.pi[x, y, action], lag_costs, transition[x, y, action, s_primes[0][0], s_primes[0][1]])
                #input('i')
        #print('self.pi',x,y,self.pi)
        #print('cost_function',const_function)
        #input('Enter...')

        self.v_m[x, y] = total

    def bellman_update_for_expert(self, old_v, x, y, cost_function, transition, unsafe_states):
        if [x, y] in self.terminal_states:
            return
        total = 0
        for action in range(self.n_actions):
            states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
            assert states[0][0] == x and states[0][1] == y
            # allow only valid action
            next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
            if not ((0<=next_state[0]<self.height) and (0<=next_state[1]<self.width)):
                continue

            x_coordinate = np.nonzero(transition[x, y, action])[0]
            y_coordinate = np.nonzero(transition[x, y, action])[1]
            for i in range(len(x_coordinate)):
                s_primes = [[x_coordinate[i], y_coordinate[i]]]
                rewards = [self.reward_mat[s_primes[0][0]][s_primes[0][1]]]
                costs = cost_function(np.array(s_primes), [action])
                orig_costs = costs # costs # (np.exp(costs)-1)
                gamma_values = self.gamma * old_v[s_primes[0][0], s_primes[0][1]]
                if [x,y] not in unsafe_states:
                    current_penalty = self.dual.nu().item()
                else:
                    current_penalty = 0
                lag_costs = self.apply_lag * current_penalty * orig_costs[0]
                total += self.pi[x, y, action] * transition[x, y, action, s_primes[0][0], s_primes[0][1]]*(rewards[0] - 0*lag_costs + gamma_values)
                #print('i, transition', x, y, action, s_primes, i, rewards[0], self.pi[x, y, action], lag_costs, transition[x, y, action, s_primes[0][0], s_primes[0][1]])
                #input('i')
        #print('self.pi',x,y,self.pi)
        #print('cost_function',const_function)
        #input('Enter...')

        self.v_m[x, y] = total


    def predict(self, obs, state, deterministic=None):
        if obs[0][0] == self.start_states[0][0] and obs[0][1] == self.start_states[0][1]:
            self.admissible_actions = self.get_actions(obs[0])
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        #print('policy_prob',obs[0][0],obs[0][1],policy_prob,self.admissible_actions)
        #input('Enter')
        best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
        action = random.choice(best_actions)
        #print('action',policy_prob,obs,self.admissible_actions,action,best_actions)
        #input('action')
        return np.asarray([action]), state

    def predict_random(self, obs, state, deterministic=None):
        """
        predict the action randomly
        """
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        print('policy_prob',obs[0][0],obs[0][1],policy_prob,self.admissible_actions)
        input('Enter')
        action = random.choice([i for i in range(len(policy_prob)) if policy_prob[i]>0 or policy_prob[i]==0])
        print('action',action)
        return np.asarray([action]), state

    def save(self, save_path):
        state_dict = dict(
            pi=self.pi,
            v_m=self.v_m,
            gamma=self.gamma,
            max_iter=self.max_iter,
            n_actions=self.n_actions,
            terminal_states=self.terminal_states,
            seed=self.seed,
            height=self.height,
            width=self.width,
            budget=self.budget,
            num_timesteps=self.num_timesteps,
            stopping_threshold=self.stopping_threshold,
        )
        torch.save(state_dict, save_path)


def load_pi(model_path, iter_msg, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)

    state_dict = torch.load(model_path)

    pi = state_dict["pi"]
    v_m = state_dict["v_m"]
    gamma = state_dict["gamma"]
    max_iter = state_dict["max_iter"]
    n_actions = state_dict["n_actions"]
    terminal_states = state_dict["terminal_states"]
    seed = state_dict["seed"]
    height = state_dict["height"]
    width = state_dict["width"]
    budget = state_dict["budget"]
    stopping_threshold = state_dict["stopping_threshold"]

    create_iteration_agent = lambda: PolicyIterationLagrange(
        env=None,
        max_iter=max_iter,
        n_actions=n_actions,
        height=height,  # table length
        width=width,  # table width
        terminal_states=terminal_states,
        stopping_threshold=stopping_threshold,
        seed=seed,
        gamma=gamma,
        budget=budget, )
    iteration_agent = create_iteration_agent()
    iteration_agent.pi = pi
    iteration_agent.v_m = v_m

    return iteration_agent
