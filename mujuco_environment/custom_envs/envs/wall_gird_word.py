import abc
import time
import numpy as np
import gym
import random
from gym.envs.mujoco import mujoco_env

from utils.data_utils import softmax
from utils.plot_utils import Plot2D
from copy import copy, deepcopy


class WallGridworld(gym.Env):
    """
    nxm Gridworld. Discrete states and actions (up/down/left/right/stay).
    Agent starts randomly.
    Goal is to reach the reward.
    Inspired from following work:
    github.com/yrlu/irl-imitation/blob/master/mdp/gridworld.py
    """

    def reset_model(self):
        pass

    def __init__(self, map_height, map_width, reward_states, terminal_states, n_actions,
                 visualization_path='./',
                 transition_prob=1.,
                 unsafe_states=[],
                 start_states=None,
                 s=None):
        """
        Construct the environment.
        Reward matrix is a 2D numpy matrix or list of lists.
        Terminal cells is a list/set of (i, j) values.
        Transition probability is the probability to execute an action and
        end up in the right next cell.
        """
        # super(WallGridworld).__init__(model_path, frame_skip)
        self.h = map_height
        self.w = map_width
        self.reward_states = reward_states
        self.reward_mat = np.zeros((self.h, self.w))
        for reward_pos in reward_states:
            self.reward_mat[reward_pos[0], reward_pos[1]] = 1
        assert (len(self.reward_mat.shape) == 2)
        # self.h, self.w = len(self.reward_mat), len(self.reward_mat[0])
        self.n = self.h * self.w
        self.terminals = terminal_states
        if n_actions == 9:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu', 8: 's'}
            self.action_space = gym.spaces.Discrete(9)
        elif n_actions == 8:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu'}
            self.action_space = gym.spaces.Discrete(8)
        elif n_actions == 4:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u'}
            self.action_space = gym.spaces.Discrete(4)
        else:
            raise EnvironmentError("Unknown number of actions {0}.".format(n_actions))
        self.transition_prob = transition_prob
        self.terminated = True
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]),
                                                high=np.array([self.h, self.w]), dtype=np.int32)
        self.unsafe_states = unsafe_states
        self.start_states = start_states
        self.steps = 0
        self.visualization_path = visualization_path
        self.reward_mat_sa = np.zeros((self.h,self.w,self.n_actions))
        self.uniform_sampling_matrix = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
        self.uniform_sampling_matrix_normalized = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
        self.expert_policy_uniform = np.zeros((self.h, self.w, self.n_actions))
        self.greedy_sampling_matrix = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
        self.greedy_sampling_matrix_normalized = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
        self.expert_policy_greedy = np.zeros((self.h,self.w,self.n_actions))
        self.active_sampling_matrix = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
        self.active_sampling_matrix_normalized = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
        self.expert_policy_active = np.zeros((self.h,self.w,self.n_actions))
        self.orig_transition = self.get_original_transition()
        self.seed(s)
        print('random_seed',s)
        # random.seed(s)

    def get_states(self):
        """
        Returns list of all states.
        """
        return filter(
            lambda x: self.reward_mat[x[0]][x[1]] not in [-np.inf, float('inf'), np.nan, float('nan')],
            [(i, j) for i in range(self.h) for j in range(self.w)]
        )

    def get_actions(self, state):
        """
        Returns list of actions that can be taken from the given state.
        """
        if self.reward_mat[state[0]][state[1]] in \
                [-np.inf, float('inf'), np.nan, float('nan')]:
            return [4]
        actions = []
        for i in range(len(self.actions)):#不用-1
            inc = self.neighbors[i]
            a = self.actions[i]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if 0 <= nei_s[0] < self.h and 0 <= nei_s[1] < self.w and \
                    self.reward_mat[nei_s[0]][nei_s[1]] not in \
                    [-np.inf, float('inf'), np.nan, float('nan')]:
                actions.append(a)
        return actions

    def terminal(self, state):
        """
        Check if the state is terminal.
        """
        for terminal_state in self.terminals:
            if state[0] == terminal_state[0] and state[1] == terminal_state[1]:
                return True
        return False

    def get_original_transition(self):
        self.orig_transition = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
        for i in range(self.h):
            for j in range(self.w):
                for k in self.get_actions([i,j]):
                    for m in self.get_next_states_and_probs([i,j],k):
                        self.orig_transition[i,j,k,m[0][0],m[0][1]] = m[1]
        for terminal_state in self.terminals:
            for k in self.get_actions(terminal_state):
                self.orig_transition[terminal_state[0],terminal_state[1],k,terminal_state[0],terminal_state[1]]=1
        # input('true')
        return self.orig_transition

    def get_reward_mat_sa(self):
        for i in range(self.h):
            for j in range(self.w):
                for action in range(self.n_actions):
                    next_state = [i+self.neighbors[action][0], j+self.neighbors[action][1]]
                    if next_state in self.reward_states:
                        self.reward_mat_sa[i,j,action] = 1
        return self.reward_mat_sa

    # eliminate invalid action
    def get_equiprobable_policy(self):
        pi = 1 / self.n_actions * np.ones((self.h, self.w, self.n_actions))
        for x in range(self.h):
            for y in range(self.w):
                if (x==0 or x==6 or y==0 or y==6) and ([x, y] not in self.terminals):
                    for action in range(self.n_actions):
                        next_state = [x+self.neighbors[action][0], y+self.neighbors[action][1]]
                        if not ((0<=next_state[0]<self.h) and (0<=next_state[1]<self.w)):
                            pi[x][y][action] = 0
                    pi[x][y] = pi[x][y] * 1/np.sum(pi[x][y])

        #print('pi',pi) 
        #input('pi')
        return pi                                 

    def get_initial_occupancy_measure(self):
        x = 1 / self.n_actions * np.ones((self.h, self.w, self.n_actions))
        for i in range(self.h):
            for j in range(self.w):
                if (i==0 or i==6 or j==0 or j==6) and ([i, j] not in self.terminals):
                    for action in range(self.n_actions):
                        next_state = [i+self.neighbors[action][0], j+self.neighbors[action][1]]
                        if not ((0<=next_state[0]<self.h) and (0<=next_state[1]<self.w)):
                            x[i][j][action] = 0
        x = x/np.sum(x)
        return x

    def get_next_states_and_probs(self, state, action):
        """
        Given a state and action, return list of (next_state, probability) pairs.
        """
        if self.terminal(state):
            return [((state[0], state[1]), 1)]
        if self.transition_prob == 1:
            inc = self.neighbors[action]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if 0 <= nei_s[0] < self.h and \
                    0 <= nei_s[1] < self.w and \
                    self.reward_mat[nei_s[0]][nei_s[1]] not in \
                    [-np.inf, float('inf'), np.nan, float('nan')]:
                return [(nei_s, 1)]
            else:
                return [((state[0], state[1]), 1)]  # state invalid
        else:
            mov_probs = np.zeros([self.n_actions])
            mov_probs[action] = self.transition_prob
            mov_probs += (1 - self.transition_prob) / self.n_actions
            for a in range(self.n_actions):
                inc = self.neighbors[a]
                nei_s = (state[0] + inc[0], state[1] + inc[1])
                if nei_s[0] < 0 or nei_s[0] >= self.h or \
                        nei_s[1] < 0 or nei_s[1] >= self.w or \
                        self.reward_mat[nei_s[0]][nei_s[1]] in \
                        [-np.inf, float('inf'), np.nan, float('nan')]:
                    # mov_probs[-1] += mov_probs[a]
                    mov_probs[a] = 0
            # sample_action = random.choices([i for i in range(self.n_actions)], weights=mov_probs, k=1)[0]
            # inc = self.neighbors[sample_action]
            # return [((state[0] + inc[0], state[1] + inc[1]), 1)]
            res = []
            mov_probs = mov_probs * 1/np.sum(mov_probs)
            for a in range(self.n_actions):
                if mov_probs[a] != 0:
                    inc = self.neighbors[a]
                    nei_s = (state[0] + inc[0], state[1] + inc[1])
                    res.append((nei_s, mov_probs[a]))
            return res

    @property
    def state(self):
        """
        Return the current state.
        """
        return self.curr_state

    def pos2idx(self, pos):
        """
        Convert column-major 2d position to 1d index.
        """
        return pos[0] + pos[1] * self.h

    def idx2pos(self, idx):
        """
        Convert 1d index to 2d column-major position.
        """
        return (idx % self.h, idx // self.h)

    def reset_with_values(self, info_dict):
        self.curr_state = info_dict['states']
        assert self.curr_state not in self.terminals
        self.terminated = False
        self.steps = 0
        return self.state

    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        if 'states' in kwargs.keys():
            self.curr_state = kwargs['states']
            assert self.curr_state not in self.terminals
            self.terminated = False
            self.steps = 0
            return self.state
        else:
            if self.start_states != None:
                random_state = random.choice(self.start_states)
                self.curr_state = random_state
            else:
                #print('random_state1',np.random.randint(self.h * self.w))
                #input('Enter..')
                random_state = np.random.randint(self.h * self.w)
                self.curr_state = self.idx2pos(random_state)
                print('random_state',random_state,self.curr_state)
                #input('Enter..')
            while self.curr_state in self.terminals or self.curr_state in self.unsafe_states:
                if self.start_states != None:
                    random_state = random.choice(self.start_states)
                    self.curr_state = random_state
                else:
                    random_state = np.random.randint(self.h * self.w)
                    self.curr_state = self.idx2pos(random_state)
            self.terminated = False
            self.steps = 0
            return self.state

    def reset_random(self, **kwargs):
        """
        Reset the environment randomly
        """
        self.start_states = None #us-code
        if 'states' in kwargs.keys():
            self.curr_state = kwargs['states']
            assert self.curr_state not in self.terminals
            self.terminated = False
            self.steps = 0
            return self.state
        else:
            if self.start_states != None:
                random_state = random.choice(self.start_states)
                self.curr_state = random_state
            else:
                #print('random_state1',np.random.randint(self.h * self.w))
                #input('Enter..')
                random_state = np.random.randint(self.h * self.w)
                self.curr_state = self.idx2pos(random_state)
                print('random_state',random_state,self.curr_state)
                #input('Enter..')
            while self.curr_state in self.terminals or self.curr_state in self.unsafe_states:
                if self.start_states != None:
                    random_state = random.choice(self.start_states)
                    self.curr_state = random_state
                else:
                    random_state = np.random.randint(self.h * self.w)
                    self.curr_state = self.idx2pos(random_state)
            self.terminated = False
            self.steps = 0
            return self.state

    # active sampling for ICRL
    def active_sampling(self, n_max, obs=[], acs=[], expert_policy=[]):
        #print('active sampling:', n_max)
        if n_max == 0:
            self.active_sampling_matrix = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
            sampling_count = np.sum(np.sum(copy(self.active_sampling_matrix),4),3)
            for i in range(self.h):
                for j in range(self.w):
                    for k in range(self.n_actions):
                        if k not in self.get_actions([i,j]):# or [i,j] in self.terminals:
                            sampling_count[i,j,k] = np.nan
            for terminal_state in self.terminals:
                #print(self.terminals)
                #input('1')
                for k in range(self.n_actions):
                    sampling_count[terminal_state[0],terminal_state[1],k]=0    
            return self.active_sampling_matrix_normalized, sampling_count, self.expert_policy_active
        else:
            # update active sampling matrix
            for num in range(len(obs)-1):
                self.active_sampling_matrix[obs[num][0]][obs[num][1]][acs[num]][obs[num+1][0]][obs[num+1][1]] += 1

        # normalize to probability
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.n_actions):
                    total_num = np.sum(self.active_sampling_matrix[i][j][k])
                    for m in range(self.h):
                        for n in range(self.w):
                            self.active_sampling_matrix_normalized[i][j][k][m][n] = self.active_sampling_matrix[i][j][k][m][n]/max(total_num,1)
        
        # update sample count for ci
        # assign np.nan to (state,action) never visited
        sampling_count = np.sum(np.sum(copy(self.active_sampling_matrix),4),3)
        #print('sampling_count', sampling_count)
        #input('sampling_count')
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.n_actions):
                    if k not in self.get_actions([i,j]): # or [i,j] in self.terminals:
                        sampling_count[i,j,k] = np.nan    

        # update estimated expert policy
        obs_unique = list(set(obs))
        for obs_num in range(len(obs_unique)):
            self.expert_policy_active[obs_unique[obs_num][0]][obs_unique[obs_num][1]] = expert_policy[obs_unique[obs_num][0]][obs_unique[obs_num][1]]

        return self.active_sampling_matrix_normalized, sampling_count, self.expert_policy_active

    # greedy sampling for ICRL
    def greedy_sampling(self, n_max, obs=[], acs=[], expert_policy=[]):
        #print('greedy sampling:', n_max)
        if n_max == 0:
            self.greedy_sampling_matrix = np.zeros((self.h,self.w,self.n_actions,self.h,self.w))
            sampling_count = np.sum(np.sum(copy(self.greedy_sampling_matrix),4),3)
            for i in range(self.h):
                for j in range(self.w):
                    for k in range(self.n_actions):
                        if k not in self.get_actions([i,j]):# or [i,j] in self.terminals:
                            sampling_count[i,j,k] = np.nan
            for terminal_state in self.terminals:
                #print(self.terminals)
                #input('1')
                for k in range(self.n_actions):
                    sampling_count[terminal_state[0],terminal_state[1],k]=0    
            return self.greedy_sampling_matrix_normalized, sampling_count, self.expert_policy_greedy
        else:
            # update greedy sampling matrix
            for num in range(len(obs)-1):
                self.greedy_sampling_matrix[obs[num][0]][obs[num][1]][acs[num]][obs[num+1][0]][obs[num+1][1]] += 1

        # normalize to probability
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.n_actions):
                    total_num = np.sum(self.greedy_sampling_matrix[i][j][k])
                    for m in range(self.h):
                        for n in range(self.w):
                            self.greedy_sampling_matrix_normalized[i][j][k][m][n] = self.greedy_sampling_matrix[i][j][k][m][n]/max(total_num,1)
        
        # update sample count for ci
        # assign np.nan to (state,action) never visited
        sampling_count = np.sum(np.sum(copy(self.greedy_sampling_matrix),4),3)
        #print('sampling_count', sampling_count)
        #input('sampling_count')
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.n_actions):
                    if k not in self.get_actions([i,j]): # or [i,j] in self.terminals:
                        sampling_count[i,j,k] = np.nan    

        # update estimated expert policy
        obs_unique = list(set(obs))
        for obs_num in range(len(obs_unique)):
            self.expert_policy_greedy[obs_unique[obs_num][0]][obs_unique[obs_num][1]] = expert_policy[obs_unique[obs_num][0]][obs_unique[obs_num][1]]

        return self.greedy_sampling_matrix_normalized, sampling_count, self.expert_policy_greedy

    # uniform sampling for ICRL
    def uniform_sampling(self, n_max):
        # random.seed(seed)
        #cnt = 0
        #print('uniform sampling:',n_max)
        #if self.uniform_sampling_matrix_normalized.any() == True:
            #input('接着更新')
        # uniform sampling


        for i in range(self.h):
            for j in range(self.w):
                self.curr_state = [i, j]
                action_list = self.get_actions(self.curr_state)
                for k in range(len(action_list)):
                    for t in range(n_max):
                        self.curr_state = [i, j]
                        random_state_next, _, _, _ = self.step(action_list[k])
                        if random.random() < 0.5:
                            self.uniform_sampling_matrix[i][j][action_list[k]][random_state_next[0]][random_state_next[1]] += 1
                        #print('params',i, j, action_list[k], random_state_next[0], random_state_next[1])
                        #input('enter')

        # normalize to probability
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.n_actions):
                    total_num = np.sum(self.uniform_sampling_matrix[i][j][k])
                    for m in range(self.h):
                        for n in range(self.w):
                            self.uniform_sampling_matrix_normalized[i][j][k][m][n] = self.uniform_sampling_matrix[i][j][k][m][n]/max(total_num,1)
            
        # assign np.nan to (state,action) never visited
        sampling_count = np.sum(np.sum(copy(self.uniform_sampling_matrix),4),3)
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.n_actions):
                    if k not in self.get_actions([i,j]) or [i,j] in self.terminals:
                        sampling_count[i,j,k] = np.nan

        # for i in range(self.h):
        #     for j in range(self.w):
        #         self.expert_policy_uniform[i][j] = expert_policy[i][j]
       
        return self.uniform_sampling_matrix_normalized, sampling_count, self.expert_policy_uniform

    def step(self, action):
        """
        Step the environment.
        """
        action = int(action)
        if self.terminal(self.state):
            self.terminated = True
            self.steps += 1
            admissible_actions = self.get_actions(self.curr_state)
            return (list(self.state),
                    0,
                    True,
                    {'x_position': self.state[0],
                     'y_position': self.state[1],
                     'admissible_actions': admissible_actions,
                     },
                    )
        self.terminated = False
        st_prob = self.get_next_states_and_probs(self.state, action)
        sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
        last_state = self.state
        next_state = st_prob[sampled_idx][0]
        reward = self.reward_mat[next_state[0]][next_state[1]]
        self.curr_state = next_state
        # return {
        #     "next_state": list(self.state),
        #     "reward": reward,
        #     "done": False,
        #     "info": {}
        # }
        self.steps += 1
        admissible_actions = self.get_actions(self.curr_state)
        return (list(self.state),
                reward,
                False,
                {'y_position': self.state[0],
                 'x_position': self.state[1],
                 'admissible_actions': admissible_actions,
                 },
                )
    def step_from_pi_expl_active(self, pi_expl,num_of_active=500):
        """
        Step the environment.
        """

        obs = []
        acs = []
        self.reset_with_values({'states':self.start_states[0]})
        #print(self.state)
        #input('self.state')
        obs.append((self.state[0], self.state[1]))
        cnt = 0
        while len(obs) < num_of_active and cnt < 2:# and self.terminal(self.state) == False:            
            if self.terminal(self.state)==True:
                cnt += 1                       
            action = np.random.choice(np.arange(0, self.n_actions), p = pi_expl[self.state[0]][self.state[1]])
            action = int(action)
            self.terminated = False
            st_prob = self.get_next_states_and_probs(self.state, action)
            sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
            last_state = self.state
            next_state = st_prob[sampled_idx][0]
            reward = self.reward_mat[next_state[0]][next_state[1]]
            self.curr_state = next_state
            obs.append(self.state)
            acs.append(action)

            self.steps += 1
            admissible_actions = self.get_actions(self.curr_state)

        return obs, acs

    def step_from_pi_expl(self, pi_expl,num_of_greedy=500):
        """
        Step the environment.
        """
        #print(random.getstate()[1][0],self.transition_prob,num_of_greedy)
        #input('transition')
        #input('current random seed')
        obs = []
        acs = []
        self.reset_with_values({'states':self.start_states[0]})
        #print(self.state)
        #input('self.state')
        obs.append((self.state[0], self.state[1]))
        cnt = 0
        while len(obs) < num_of_greedy and cnt < 2:# and self.terminal(self.state) == False:            
            if self.terminal(self.state)==True:
                cnt += 1                       
            action = np.random.choice(np.arange(0, self.n_actions), p = pi_expl[self.state[0]][self.state[1]])
            action = int(action)
            self.terminated = False
            st_prob = self.get_next_states_and_probs(self.state, action)
            sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
            last_state = self.state
            next_state = st_prob[sampled_idx][0]
            reward = self.reward_mat[next_state[0]][next_state[1]]
            self.curr_state = next_state
            obs.append(self.state)
            acs.append(action)

            self.steps += 1
            admissible_actions = self.get_actions(self.curr_state)

        return obs, acs

    def step_from_pi_expl_UCB(self, pi_expl, reward_state_action, num_of_greedy=500):
        """
        Step the environment.
        """
        #print(random.getstate()[1][0],self.transition_prob,num_of_greedy)
        #input('transition')
        #input('current random seed')
        obs = []
        acs = []
        self.reset_with_values({'states':self.start_states[0]})
        #print(self.state)
        #input('self.state')
        obs.append((self.state[0], self.state[1]))
        cnt = 0
        while len(obs) < num_of_greedy and cnt < 2:# and self.terminal(self.state) == False:            
            if self.terminal(self.state)==True:
                #print(cnt, reward)
                #input('cnt')
                cnt += 1                       
            action = np.random.choice(np.arange(0, self.n_actions), p = pi_expl[self.state[0]][self.state[1]])
            action = int(action)
            self.terminated = False
            st_prob = self.get_next_states_and_probs(self.state, action)
            sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
            last_state = self.state
            next_state = st_prob[sampled_idx][0]
            reward = self.reward_mat[next_state[0]][next_state[1]]
            if reward == 1:
                reward_state_action.append([last_state[0], last_state[1], action])
                #print(last_state[0], last_state[1], action, reward)
                #input('UCB')
            self.curr_state = next_state
            obs.append(self.state)
            acs.append(action)

            self.steps += 1
            admissible_actions = self.get_actions(self.curr_state)

        return obs, acs, reward_state_action
            
    def step_from_us(self, action):
        """
        Step the environment.
        """
        #print('step2')
        action = int(action)
        if self.terminal(self.state):
            self.terminated = True
            self.steps += 1
            admissible_actions = self.get_actions(self.curr_state)
            return (list(self.state),
                    0,
                    True,
                    {'x_position': self.state[0],
                     'y_position': self.state[1],
                     'admissible_actions': admissible_actions,
                     },
                    )
        self.terminated = False
        st_prob = self.get_next_states_and_probs(self.state, action)
        sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
        last_state = self.state
        next_state = st_prob[sampled_idx][0]
        reward = self.reward_mat[next_state[0]][next_state[1]]
        #print('st_prob',st_prob)
        self.curr_state = next_state
        # return {
        #     "next_state": list(self.state),
        #     "reward": reward,
        #     "done": False,
        #     "info": {}
        # }
        self.steps += 1
        admissible_actions = self.get_actions(self.curr_state)
        return (list(self.state),
                reward,
                False,
                {'y_position': self.state[0],
                 'x_position': self.state[1],
                 'admissible_actions': admissible_actions,
                 },
                )

    def seed(self, s=None):
        """
        Seed this environment.
        """
        random.seed(s)
        np.random.seed(s)

    def render(self, mode, **kwargs):
        """
        Render the environment.
        """
        self.state_mat = np.zeros([self.h, self.w, 3])
        self.state_mat[self.state[0], self.state[1], :] = 1.
        if not hasattr(self, "plot"):
            self.plot = Plot2D({
                "env": lambda p, l, t: self,
            }, [
                [
                    lambda p, l, t: not l["env"].terminated,
                    lambda p, l, o, t: p.imshow(l["env"].state_mat, o=o)
                ],
            ], mode="dynamic", interval=1)
        self.plot.show(block=False)

        # if "mode" in kwargs.keys() and kwargs["mode"] == "rgb_array":
        if mode == "rgb_array":
            self.plot.fig.canvas.draw()
            img = np.frombuffer(self.plot.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.plot.fig.canvas.get_width_height()[::-1] + (3,))
            return img

