# Understanding Constraint Inference in Safety-Critical Inverse Reinforcement Learning

This is the repo for the paper: Understanding Constraint Inference in Safety-Critical Inverse Reinforcement Learning, published at ICLR 2025. Note that:

## Contribution
### 1. Core Problem: Inverse Constraint Inference (ICI)
- In many practical applications, constraints are not readily available, so we need to infer the constraints followed by expert agents based on their behaviors. This is known as the ICI problem. The input of an ICI problem is the environment with known reward signals and the expert policy. Typically, the environment is a Markov Decision Process and the expert policy is optimal. The output of an ICI problem is constraint signals based on which an agent can align with expert behaviors.
### 2. Critical Question
- *Can we implicitly embed constraint signals into reward functions and effectively solve ICI problem using a classic reward inference algorithm?*
### 3. Solver One: Inverse Reward Correction (IRC)
- An IRC solver outputs the constraint signal as reward correction terms $\Delta r$ so that $r^c(s,a)=r(s,a)+{\mathit{\Delta r}}(s,a)$ ensures alignment with expert behaviors.
### 4. Solver Two: Inverse Constrained Reinforcement Learning (ICRL)
- An ICRL solver outputs the constraint signal as cost functions $c$ so that the constraint condition ensures alignment with expert behaviors.
### 5. Key Insight
- Training efficiency: IRC > ICRL
- Cross-environment Transferability: ICRL > IRC

## Setup Experimental Environments 
### 1. Create Python Environment 
1. Please install the conda before proceeding.
2. Create a conda environment and install the packages:
   
```
mkdir save_model
mkdir evaluate_model
conda env create -n ircvsicrl python=3.9 -f python_environment.yml
conda activate ircvsicrl
```
You can also first install Python 3.9 with the torch (2.0.1+cu117) and then install the packages listed in `python_environment.yml`.

### 2. Setup MuJoCo Environment (you can also refer to [MuJoCo Setup](https://github.com/Guiliang/ICRL-benchmarks-public/blob/main/virtual_env_tutorial.md))
1. Download the MuJoCo version 2.1 binaries for Linux or OSX.
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
3. Install and use mujoco-py.
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment

export MUJOCO_PY_MUJOCO_PATH=YOUR_MUJOCO_DIR/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_MUJOCO_DIR/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
## Experiments
### 1. IRC vs ICRL
```
# step into the interface dir
cd ./interface/
# train ICRL in source
python train_icrl.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1_target.yaml
# transfer ICRL to target
python train_icrl_target.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1_target.yaml
# train IRC in source
python train_irc.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1_target.yaml
# transfer IRC to target
python train_irc_target.py ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1.yaml ../config/mujoco_WGW-discrete-v0/train_ICRL_discrete_WGW-v0-setting1_target.yaml
```

### 2. Numerical Validation of Theorem 5.3
Please refer to `numerical_analysis_of_example_in_Fig1.ipynb`.

## Related works

### 1. Related to ICRL solver

- [Benchmarking constraint inference in inverse reinforcement learning](https://arxiv.org/pdf/2206.09670) [ICLR 2023]
- [A Comprehensive Survey on Inverse Constrained Reinforcement Learning: Definitions, Progress and Challenges](https://openreview.net/pdf?id=WUQsBiJqyP) [TMLR 2025]

### 2. Related to IRC solver

- [Mind the Gap: Offline Policy Optimization for Imperfect Rewards](https://openreview.net/forum?id=WumysvcMvV6) [ICLR 2023]
- [Simplifying constraint inference with inverse reinforcement learning](https://openreview.net/pdf?id=T5Cerv7PT2) [NeurIPS 2024]

### 3. Related to Constraint Inference
- [Awesome-Constraint-Inference-in-RL](https://github.com/Jasonxu1225/Awesome-Constraint-Inference-in-RL)
- [Constrained-Decision-Making-Paper-List](https://github.com/zbzhu99/Constrained-Decision-Making-Paper-List)

## Citation

Should you find this work helpful, please consider citing:
```
@inproceedings{
anonymous2025understanding,
title={Understanding Constraint Inference in Safety-Critical Inverse Reinforcement Learning},
author={Anonymous},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=B2RXwASSpy}
}
```

## Acknowledgement
1. The experimental environment is mainly based on the [MuJoCo](https://mujoco.org/).
2. The implementation is based on the code from [ICRL-benchmark](https://github.com/Guiliang/ICRL-benchmarks-public/tree/main).

