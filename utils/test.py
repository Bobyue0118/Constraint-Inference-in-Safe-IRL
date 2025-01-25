import torch
import numpy as np
import random
from copy import deepcopy


# def get_net_arch(config):
#     """
#     Returns a dictionary with sizes of layers in policy network,
#     value network and cost value network.
#     """
#     try:
#         separate_layers = dict(pi=config.policy_layers,  # Policy Layers
#                                vf=config.reward_vf_layers,  # Value Function Layers
#                                cvf=config.cost_vf_layers)  # Cost Value Function Layers
#     except:
#         print("Could not define layers for policy, value func and " + \
#               "cost_value_function, will attempt to just define " + \
#               "policy and value func")
#         separate_layers = dict(pi=config.policy_layers,  # Policy Layers
#                                vf=config.reward_vf_layers)  # Value Function Layers
#
#     if config.shared_layers is not None:
#         return [*config.shared_layers, separate_layers]
#     else:
#         return [separate_layers]

def get_net_arch(config, log_file):
    """
    Returns a dictionary with sizes of layers in policy network,
    value network and cost value network.
    """

    if 'cost_vf_layers' in config['PPO'].keys():
        separate_layers = dict(pi=config['PPO']['policy_layers'],  # Policy Layers
                               vf=config['PPO']['reward_vf_layers'],  # Value Function Layers
                               cvf=config['PPO']['cost_vf_layers'])  # Cost Value Function Layers
    else:
        separate_layers = dict(pi=config['PPO']['policy_layers'],  # Policy Layers
                               vf=config['PPO']['reward_vf_layers'])  # Value Function Layers

    print("PPO layers are:", separate_layers, flush=True, file=log_file)
    return [separate_layers]


def handle_model_parameters(model, active_keywords, model_name, log_file, set_require_grad):
    """determine which parameters should be fixed"""
    # exclude some parameters from optimizer
    param_frozen_list = []  # should be changed into torch.nn.ParameterList()
    param_active_list = []  # should be changed into torch.nn.ParameterList()
    fixed_parameters_keys = []
    active_parameters_keys = []
    parameters_info = []

    for k, v in model.named_parameters():
        keep_this = False
        size = torch.numel(v)
        parameters_info.append("{0}:{1}".format(k, size))
        for keyword in active_keywords:
            if keyword in k:
                param_active_list.append(v)
                active_parameters_keys.append(k)
                keep_this = True
                break
        if not keep_this:
            param_frozen_list.append(v)
            if set_require_grad:
                v.requires_grad = False  # fix the parameters https://pytorch.org/docs/master/notes/autograd.html
            fixed_parameters_keys.append(k)

    print('-' * 30 + '{0} Optimizer'.format(model_name) + '-' * 30, file=log_file, flush=True)
    print("Active parameters are: {0}".format(str(active_parameters_keys)), file=log_file, flush=True)
    print("Fixed parameters are: {0}".format(str(fixed_parameters_keys)), file=log_file, flush=True)
    # print(parameters_info, file=log_file, flush=True)
    param_frozen_list = torch.nn.ParameterList(param_frozen_list)
    param_active_list = torch.nn.ParameterList(param_active_list)
    print('-' * 60, file=log_file, flush=True)

    return param_frozen_list, param_active_list


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def stability_loss(input_data, aggregates, concepts, relevances):
    """Computes Robustness Loss for the Compas data

    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design
    Parameters
    ----------
    input_data   : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)

    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    batch_size = input_data.size(0)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes).to(input_data.device)
    J_yx = torch.autograd.grad(outputs=aggregates,
                               inputs=input_data,
                               grad_outputs=grad_tensor,
                               create_graph=True,
                               only_inputs=True)[0]
    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1)

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances
    robustness_loss = robustness_loss.norm(p='fro', dim=1)
    return robustness_loss


def dirichlet_kl_divergence_loss(alpha, prior):
    """
    KL divergence between two dirichlet distribution
    The mean is alpha/(alpha+beta) and variance is alpha*beta/(alpha+beta)^2*(alpha+beta+1)
    There are multiple ways of modelling a dirichlet:
    1) by Laplace approximation with logistic normal: https://arxiv.org/pdf/1703.01488.pdf
    2) by directly modelling dirichlet parameters: https://arxiv.org/pdf/1901.02739.pdf
    code reference：
    1） https://github.com/sophieburkhardt/dirichlet-vae-topic-models
    2） https://github.com/is0383kk/Dirichlet-VAE
    """
    analytical_kld = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(prior, dim=1))
    analytical_kld += torch.sum(torch.lgamma(prior), dim=1)
    analytical_kld -= torch.sum(torch.lgamma(alpha), dim=1)
    minus_term = alpha - prior
    # tmp = torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    digamma_term = torch.digamma(alpha) - \
                   torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    test = torch.sum(torch.mul(minus_term, digamma_term), dim=1)
    analytical_kld += test
    # self.analytical_kld = self.mask * self.analytical_kld  # mask paddings
    return analytical_kld


def torch_kron_prod(a, b):
    """
    :param a: matrix1 of size [b, M]
    :param b: matrix2 of size [b, N]
    :return: matrix of size [b, M, N]
    """
    res = torch.einsum('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res


def load_policy_iteration_config(config, env_configs, train_env, seed, log_file):
    pi_parameters = {
        "env": train_env,
        "seed": seed,
        "stopping_threshold": config["iteration"]["stopping_threshold"],
        "max_iter": config["iteration"]["max_iter"],
        "gamma": config["iteration"]["gamma"],
        "n_actions": env_configs['n_actions'],
        "height": env_configs['map_height'],
        "width": env_configs['map_width'],
        "start_states": env_configs['start_states'],
        "terminal_states": env_configs['terminal_states'],
        "penalty_initial_value": config['iteration']['penalty_initial_value'],
        "penalty_learning_rate": config['iteration']['penalty_learning_rate'],
        "log_file": log_file,
        "reward_states": env_configs['reward_states'],

    }
    pi_parameters.update({"penalty_min_value": config['iteration']['nu_min_clamp']})
    pi_parameters.update({"penalty_max_value": config['iteration']['nu_max_clamp']})
    return pi_parameters


def load_ppo_config(config, train_env, seed, log_file):
    ppo_parameters = {
        "policy": config['PPO']['policy_name'],
        "env": train_env,
        "learning_rate": config['PPO']['learning_rate'],
        "n_steps": config['PPO']['n_steps'],
        "batch_size": config['PPO']['batch_size'],
        "n_epochs": config['PPO']['n_epochs'],
        "clip_range": config['PPO']['clip_range'],
        "ent_coef": config['PPO']['ent_coef'],
        "max_grad_norm": config['PPO']['max_grad_norm'],
        "use_sde": config['PPO']['use_sde'],
        "sde_sample_freq": config['PPO']['sde_sample_freq'],
        "target_kl": config['PPO']['target_kl'],
        "verbose": config['verbose'],
        "seed": seed,
        "device": config['device'],
        "policy_kwargs": dict(net_arch=get_net_arch(config, log_file))
    }
    if config["group"] == "PPO" or config["group"] == "GAIL":
        ppo_parameters.update({
            "gamma": config['PPO']['reward_gamma'],
            "gae_lambda": config['PPO']['reward_gae_lambda'],
            "vf_coef": config['PPO']['reward_vf_coef'],
        })
    elif config['group'] == "PPO-Lag" or config['group'] == "Binary" or config['group'] == "ICRL" or config[
        'group'] == "VICRL":
        # elif config['group'] == "PPO-Lag":
        ppo_parameters.update({
            "reward_gamma": config['PPO']['reward_gamma'],
            "reward_gae_lambda": config['PPO']['reward_gae_lambda'],
            "cost_gamma": config['PPO']['cost_gamma'],
            "cost_gae_lambda": config['PPO']['cost_gae_lambda'],
            "clip_range_reward_vf": config['PPO']['clip_range_reward_vf'],
            "clip_range_cost_vf": config['PPO']['clip_range_cost_vf'],
            "reward_vf_coef": config['PPO']['reward_vf_coef'],
            "cost_vf_coef": config['PPO']['cost_vf_coef'],
            "penalty_initial_value": config['PPO']['penalty_initial_value'],
            "penalty_learning_rate": config['PPO']['penalty_learning_rate'],
            "budget": config['PPO']['budget'],
            "pid_kwargs": dict(alpha=config['PPO']['budget'],
                               penalty_init=config['PPO']['penalty_initial_value'],
                               Kp=config['PPO']['proportional_control_coeff'],
                               Ki=config['PPO']['integral_control_coeff'],
                               Kd=config['PPO']['derivative_control_coeff'],
                               pid_delay=config['PPO']['pid_delay'],
                               delta_p_ema_alpha=config['PPO']['proportional_cost_ema_alpha'],
                               delta_d_ema_alpha=config['PPO']['derivative_cost_ema_alpha'], ),
        })
    else:
        raise ValueError("Unknown Group {0}".format(config['group']))

    return ppo_parameters

def get_hoeffding_ci_us(height, width, n_actions, sample_count, v_m, zeta_max, gamma, epsilon, delta=0.01):
    n_states = height*width
    sample_count = np.maximum(sample_count, 1)
    r_max = 1
    #print('sample_count1',sample_count)
    #input('in')
    ci = np.sqrt(
        np.log(36 * n_states * n_actions * np.square(sample_count) / delta)
        / (2*sample_count)
    )
    #print('ci',np.round(ci,1))
    #input('in2')
    v_m = np.repeat(v_m, n_actions).reshape(height, width, n_actions)
    #ci *= gamma * (2*zeta_max* np.max(v_m)/(1-gamma)  + epsilon)
    ci *= gamma * (2*zeta_max*r_max /(1-gamma)**2  + epsilon)
    #print('ci',np.round(ci,1))
    #input('in3')
    return ci

def get_hoeffding_ci_active(height, width, n_actions, sample_count, v_m, zeta_max, gamma, epsilon, delta=0.1):
    n_states = height*width
    sample_count = np.maximum(sample_count, 1)
    r_max = 1
    #print('sample_count1',sample_count)
    #input('in')
    ci = np.sqrt(
        np.log(36 * n_states * n_actions * np.square(sample_count) / delta)
        / (2*sample_count)
    )
    #print('ci',np.round(ci,1))
    #input('in2')
    #v_m = np.repeat(v_m, n_actions).reshape(height, width, n_actions)
    #ci *= gamma * ((zeta_max/(1-gamma))*(2*r_max/(1-gamma)+r_max*(1+gamma)/(1-gamma))  + epsilon)
    ci *= gamma * (((3+gamma)/(1-gamma)**2)*r_max*zeta_max  + epsilon)
    #print('ci',np.round(ci,1))
    #input('in3')
    return ci

def get_hoeffding_ci_greedy(height, width, n_actions, sample_count, v_m, zeta_max, gamma, epsilon, delta=0.1):
    n_states = height*width
    sample_count = np.maximum(sample_count, 1)
    r_max = 1
    #print('sample_count1',sample_count)
    #input('in')
    ci = np.sqrt(
        np.log(36 * n_states * n_actions * np.square(sample_count) / delta)
        / (2*sample_count)
    )
    #print('ci',np.round(ci,1))
    #input('in2')
    v_m = np.repeat(v_m, n_actions).reshape(height, width, n_actions)
    #ci *= gamma * ((zeta_max/(1-gamma))*(2*np.max(v_m)+r_max*(1+gamma)/(1-gamma))  + epsilon)
    ci *= gamma * (((3+gamma)/(1-gamma)**2)*r_max*zeta_max  + epsilon)
    #print('ci',np.round(ci,1))
    #input('in3')
    return ci

def valueIteration(height, width, ci, n_actions, gamma, transition, env, stopping_threshold):
    #print(random.getstate()[1][0])
    #input('current random seed1')
    v = np.inf*np.ones((height, width))
    v_prime = np.zeros((height, width))
    pi = np.zeros((height, width, n_actions))
    #print("During the value iteration:\n")
    
    while True:
        error = 0
        v = deepcopy(v_prime)
        #print('v',np.round(v,2))
        #input('v')
        #print(transition[6][6][0][6][6],ci[6][6][0])
        #input('v')
        pi = np.zeros((height, width, n_actions))
        for i in range(height):
            for j in range(width):
                v_list=[]
                v_list_action=[]
                for action in env.get_actions([i,j]):
                    v_ = ci[i][j][action]
                    v_list_action.append(action)
                    #print(ci)
                    #input('1')
                    for m in env.get_next_states_and_probs([i,j], action):
                        v_ += gamma*transition[i][j][action][m[0][0]][m[0][1]]*v[m[0][0]][m[0][1]]
                        #print(v_,m,transition[i][j][action][m[0][0]][m[0][1]],v[m[0][0]][m[0][1]])
                        #input('2')
                    v_list.append(v_)
                    #print(v_list)
                    #input('3')
                #print(v_list)
                #input('4')
                v_prime[i][j] = max(v_list) # Bellman update
                #print(v_prime[i][j])
                #input('5')
                best_action = [v_list_action[index] for index, value in enumerate(v_list) if value == max(v_list)]
                #print(best_action)
                #input('best_action')
                for k in best_action:
                    pi[i][j][k] = 1/len(best_action)
                error = max(error, abs(v_prime[i][j]-v[i][j]))
                #v_prime[reward_states[0][0],reward_states[0][1]]=np.max(v_prime)
                #print(error)
                #input('error')
                
        if error < stopping_threshold:
            #print('v',np.round(v,2))
            #print('pi',pi)
            #input('v')            
            break
    return pi

def costValueIteration(height, width, ci, n_actions, gamma, transition, env, stopping_threshold):
    v = np.inf*np.ones((height, width))
    v_prime = np.zeros((height, width))
    pi = np.zeros((height, width, n_actions))
    #print("During the value iteration:\n")
    
    while True:
        error = 0
        v = deepcopy(v_prime)
        #print('v',np.round(v,2))
        #input('v')
        #print(transition[6][6][0][6][6],ci[6][6][0])
        #input('v')
        pi = np.zeros((height, width, n_actions))
        for i in range(height):
            for j in range(width):
                v_list=[]
                v_list_action=[]
                for action in env.get_actions([i,j]):
                    v_ = ci[i][j][action]
                    v_list_action.append(action)
                    #print(ci)
                    #input('1')
                    for m in env.get_next_states_and_probs([i,j], action):
                        v_ += gamma*transition[i][j][action][m[0][0]][m[0][1]]*v[m[0][0]][m[0][1]]
                        #print(v_,m,transition[i][j][action][m[0][0]][m[0][1]],v[m[0][0]][m[0][1]])
                        #input('2')
                    v_list.append(v_)
                    #print(v_list)
                    #input('3')
                #print(v_list)
                #input('4')
                v_prime[i][j] = max(v_list) # Bellman update
                #print(v_prime[i][j])
                #input('5')
                best_action = [v_list_action[index] for index, value in enumerate(v_list) if value == max(v_list)]
                #print(best_action)
                #input('best_action')
                for k in best_action:
                    pi[i][j][k] = 1/len(best_action)
                error = max(error, abs(v_prime[i][j]-v[i][j]))
                #v_prime[reward_states[0][0],reward_states[0][1]]=np.max(v_prime)
                #print(error)
                #input('error')
                
        if error < stopping_threshold:
            #print('v',np.round(v,2))
            #print('pi',pi)
            #input('v')            
            break
    return v_prime

def cal_GIoU(a, b):
    #a = np.array([[0,0,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]])
    #b = np.array([[0,0,0,0],[0,1,1,0],[0,0,0,0],[0,1,1,0]])
    #a[a>0] = 1
    #b[b>0] = 1
    #print(a,np.round(b,2))
    #print(np.logical_or(a,b).astype(int))
    #assert a.shape==b.shape, "shape of a and b does not match!"
    c = np.zeros(a.shape)
    positive_b = b[b>0]
    positive_b_min = min(np.min(positive_b),1) if len(positive_b)>0 else 1
    if np.sum(a*b)>0:
        #GIoU = np.sum(a*b/positive_b_min)/np.sum(np.maximum(a,b/positive_b_min))# weighted GIoU
        print(np.round(a/positive_b_min,2),np.round(b/positive_b_min,2),np.round(a/positive_b_min*b/positive_b_min,2), np.round(np.maximum(a/positive_b_min,b/positive_b_min,a/positive_b_min*b/positive_b_min),2))
        GIoU = np.sum(a/positive_b_min*b/positive_b_min)/np.sum(np.maximum(np.maximum(a/positive_b_min,b/positive_b_min),a/positive_b_min*b/positive_b_min))# weighted GIoU
    else:
        #GIoU = np.exp(-np.sum(np.maximum(a,b/positive_b_min)))-1
        GIoU = np.exp(-np.sum(np.maximum(a/positive_b_min,b/positive_b_min)))-1
        #if not np.all(b==0):#b中存在元素不为0
            #nonzero_indices_a = np.nonzero(a)
            #nonzero_indices_b = np.nonzero(b) 
            #print(nonzero_indices_a[0],nonzero_indices_a[1])
            #min_x = min(np.min(nonzero_indices_a[0]), np.min(nonzero_indices_b[0]))
            #min_y = min(np.min(nonzero_indices_a[1]), np.min(nonzero_indices_b[1]))
            #max_x = max(np.max(nonzero_indices_a[0]), np.max(nonzero_indices_b[0]))
            #max_y = max(np.max(nonzero_indices_a[1]), np.max(nonzero_indices_b[1]))
            #print(min_x,min_y,max_x,max_y)
            #for i in range(min_x, max_x+1):
                #for j in range(min_y, max_y+1):
                    #c[i,j] = 1
        #else: #b中所有元素都是0
            #c = deepcopy(a)
        #print(c)
        #GIoU = -np.sum(c-np.logical_or(a,b).astype(int))/np.sum(c)
    print(GIoU)
    #input('GIoU1')
    return GIoU

def cal_discounted_cumulative_rewards(traj, reward_states, gamma):
    rewards = 0
    for i in range(len(traj)):
        if traj[i] in reward_states:
            rewards += gamma**(i+1) * 1
    return rewards

def cal_discounted_cumulative_costs(traj, unsafe_states, gamma):
    costs = 0
    for i in range(len(traj)):
        if traj[i] in unsafe_states:
            costs += gamma**(i+1) * 1
    return costs

if __name__ == "__main__":
    a = np.array( [[0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 1., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 1., 1., 1., 1., 1.],
 [0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0.]])
    b = np.array( [[ 0.,    0.,    0.,    0.,    0.,    0.,    0.  ],
 [ 0.,    0.,    0.,    0.,    0.,    0.,    0.  ],
 [ 1.27,  1.87,  1.8,   1.66,  0.67,  0.,    0.  ],
 [ 0.,    0.,    0.,    0.,    0.,    0.,    0.  ],
 [ 0.,    0.,    2.43,  5.63, 11.01, 12.25,  8.48],
 [ 0.,    0.,    0.,    0.,    0.,    0.,    0.  ],
 [ 0.,    0.,    0.,    0.,    0.,    0.,    0.  ]])
    print(cal_GIoU(a,b))
