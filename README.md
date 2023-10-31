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
from dqn import DQN
from ppo import PPO
from ppo_discrete import PPO_Discrete

import argparse
```

* The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments 
it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically 
generates help and usage messages. The module will also issue errors when users give the program invalid arguments.
Click [Python 学习之 argparse 模块](https://zhuanlan.zhihu.com/p/28871131), [argparse 模块用法实例详解](https://zhuanlan.zhihu.com/p/56922793) 
or [argparse 教程 | PYthon](https://docs.python.org/zh-cn/3/howto/argparse.html) to know more details.

* We can simplify the args parsing procedure into four steps:
  1. `import argparse` 
  2. `parser = argparse.ArgumentParser()`
  3. `parser.add_argument()`
  4. `parser.parse_args()`

After getting the `args`, the class named by RL algorithm takes args as input and output an instantiated policy.

```python
args = parser.parse_args()

if args.method == 'ppo':
    policy = PPO(args)
elif args.method == 'ppo_d':
    policy = PPO_Discrete(args)
elif args.method == 'dqn':
    policy = DQN(args)
```

Lastly, `policy.run()` was executed in an endless loop.

Now let's turn to `ppo.py`

## ppo.py

There are two classes defined in the `ppo.py`: one is `Net`, where the network architecture is defined, the other is `PPO`, 
where function `make_data`, `update`, `run` is defined.

In the last session, the PPO takes the variable `args` as input parameters. And let's see how it works in PPO:

```python
def __init__(self, args):
    self.args = args

    # initialise parameters
    self.env = Cartpole(args)

    self.mini_batch_size = self.args.num_envs * self.mini_chunk_size

    self.net = Net(self.env.num_obs, self.env.num_act).to(args.sim_device)
    self.action_var = torch.full((self.env.num_act,), 0.1).to(args.sim_device)
```

In the `__init__` function, an RL environment was initiated with the input parameters `args`.
A network `self.net` and an Adam optimizer `self.optim` are also initiated.

`PPO.run()` 中主要负责下述内容
* 获取观测
  * `obs = self.env.obs_buf.clone()`
* 策略网络通过观测计算动作 
  * `mu = self.net.pi(obs)`
  * `dist = MultivariateNormal(mu, cov_mat)`
  * `action = dist.sample()`
* 在强化学习环境中采取动作并获取相应的状态 / 观测、奖励等 
  * `self.env.step(action)`
  * `next_obs, reward, done = self.env.obs_buf.clone(), self.env.reward_buf.clone(), self.env.reset_buf.clone()`
  * `self.env.reset()`
* 储存新数据
  * `self.data.append((obs, action, reward, next_obs, log_prob, 1 - done))`
* 当数据量不小于 `rollout_size` 时，更新网络 
  * `self.update()`
* 计数器自增
  * `self.run_step += 1`




