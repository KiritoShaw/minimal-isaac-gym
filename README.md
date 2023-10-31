# Minimal Isaac Gym
This repository provides a minimal example of NVIDIA's [Isaac Gym](https://developer.nvidia.com/isaac-gym), to assist other researchers like me to **quickly understand the code structure**, to be able to design fully customised large-scale reinforcement learning experiments.

The example is based on the official implementation from the Isaac Gym's [Benchmark Experiments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), for which we have followed a similar implementation on [Cartpole](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/cartpole.py), but with a minimal number of lines of code aiming for maximal readability, and without using any third-party RL frameworks. 

**Note**: The current implementation is based on Isaac Gym Preview Version 3, with the support for two RL algorithms: *DQN* and *PPO* (both continuous and discrete version). PPO seems to be the default RL algorithm for Isaac Gym from the recent works of [Learning to walk](https://arxiv.org/abs/2109.11978) and [Object Re-orientation](https://arxiv.org/abs/2111.03043), since it **only requires on-policy training data** and therefore to make it a much simpler implementation coupled with Isaac Gym's APIs. 

*Both DQN and PPO are expected to converge under 1 minute.*

## Usage
Simply run `python trainer.py --method {dqn; ppo, ppo_d}`.

## Disclaimer
I am also very new to Isaac Gym, and I cannot guarantee my implementation is absolutely correct. If you have found anything unusual or unclear that can be improved, PR or Issues are highly welcomed.

# Notes by Shaw_ZYX

## trainer.py

It's a simpler version of `legged_gym/scripts/train.py` of legged gym.

``` python
import argparse
```

* The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments 
it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically 
generates help and usage messages. The module will also issue errors when users give the program invalid arguments.
Click [argparse 模块用法实例详解](https://zhuanlan.zhihu.com/p/56922793) and [argparse 教程 | PYthon](https://docs.python.org/zh-cn/3/howto/argparse.html) to know more details




