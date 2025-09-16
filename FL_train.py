from ast import arg
import csv
import datetime
import pprint
import re
from tkinter import font
import copy

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# from test_xp_harq import NOMA_Environ
from FL_test_noma_harq_env_copy import NOMA_Environ
from model_copy import ActorNet, CriticNet

import json
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.exploration import GaussianNoise
from tianshou.data import Batch, VectorReplayBuffer, ReplayBuffer, Collector, PrioritizedReplayBuffer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.policy import TD3Policy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use gpu or not')
parser.add_argument('--gpu_fraction', default=(0.5, 0), help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument("--task", type=str, default="TD3-Power-outage-train")
parser.add_argument('--seed', type=int, default=123, help='Value of random seed')
parser.add_argument('--reward-threshold', type=float, default=3000000)
parser.add_argument('--buffer-size', type=int, default=20000)
parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128])
parser.add_argument('--actor-lr', type=float, default=1e-8)
parser.add_argument('--critic-lr', type=float, default=1e-8)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--exploration-noise", type=float, default=1)
parser.add_argument("--policy-noise", type=float, default=0.3)
parser.add_argument("--noise-clip", type=float, default=0.6)
parser.add_argument("--update-actor-freq", type=int, default=2)
parser.add_argument("--start-timesteps", type=int, default=20000)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--step-per-epoch", type=int, default=50000)
parser.add_argument('--slot-per-test', type=int, default=6000)
parser.add_argument("--step-per-collect", type=int, default=10)
parser.add_argument("--update-per-step", type=int, default=0.1)
parser.add_argument('--episode-per-test', type=int, default=1)
parser.add_argument("--n-step", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--training-num", type=int, default=1)
parser.add_argument("--test-num", type=int, default=1)
parser.add_argument("--logdir", type=str, default="log")
parser.add_argument("--render", type=float, default=0.)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--resume-path", type=str, default=None)
parser.add_argument("--resume-id", type=str, default=None)
parser.add_argument(
    "--logger",
    type=str,
    default="tensorboard",
    choices=["tensorboard", "wandb"],
)
parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
parser.add_argument(
    "--watch",
    default=False,
    action="store_true",
    help="watch the play of pre-trained policy only",
)

def average(dict_list):
    # 初始化一个空字典
    avg_dict = {}
    # 获取列表中的第一个字典的键
    keys = dict_list[0].keys()
    # 对每个键进行遍历
    for key in keys:
        # 初始化一个空列表，用来存储每个字典中该键对应的值
        values = []
        # 对每个字典进行遍历
        for d in dict_list:
            # 将该键对应的值添加到列表中
            values.append(d[key])
        # 计算列表中的值的平均值，作为新字典中该键对应的值
        avg_dict[key] = sum(values) / len(values)
    # 返回新字典
    return avg_dict


def args_read(args_path):
    args = argparse.ArgumentParser()
    args_dict = vars(args)

    with open(args_path, 'rt') as f:
        args_dict.update(json.load(f))
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.is_available())

    """
    XP-PER-DDPG-power测试
    """

    args = parser.parse_args()

    setup_seed(args.seed)
    phi=10**-2
    max_K = 2
    env = NOMA_Environ(35, max_K, args.seed, phi=phi, beta=[8.5, 8.5], training=True)
    train_envs = NOMA_Environ(35, max_K, args.seed, training=True)
    test_envs = NOMA_Environ(35, max_K, args.seed, training=False)
    args.state_shape = env.observation_space[env.agents[0]].shape or env.observation_space[env.agents[0]].n
    args.action_shape = env.action_space[env.agents[0]].shape or env.action_space[env.agents[0]].n
    args.max_action = env.action_space[env.agents[0]].high[0]
    agent_nums = len(env.agents)
    agents = []
    buffers = []
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space[env.agents[0]].low), np.max(env.action_space[env.agents[0]].high))

    # model
    actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=args.max_action)
    actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    critic1_net = CriticNet(args.state_shape, args.action_shape, "critic1")
    critic1_optim = torch.optim.Adam(critic1_net.parameters(), lr=args.critic_lr)
    critic2_net = CriticNet(args.state_shape, args.action_shape, "critic2")
    critic2_optim = torch.optim.Adam(critic2_net.parameters(), lr=args.critic_lr)


    # 创建策略
    agent = TD3Policy(actor_net,
                    actor_optim,
                    critic1_net,
                    critic1_optim,
                    critic2_net,
                    critic2_optim,
                    tau=args.tau,
                    gamma=args.gamma,
                    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
                    policy_noise=args.policy_noise,
                    update_actor_freq=args.update_actor_freq,
                    noise_clip=args.noise_clip,
                    estimation_step=args.n_step,
                    action_space=env.action_space,
                    )


    buffer = PrioritizedReplayBuffer(args.buffer_size, 0.5, 0.5)
    # buffer = ReplayBuffer(args.buffer_size)
    w = [0.25, 0.75]
    for i in range(agent_nums):
        agents.append(copy.deepcopy(agent))
        agents[i] = agents[i].train()
        buffers.append(copy.deepcopy(buffer))

    global_agent = copy.deepcopy(agent)
    
    max_rew = 0 
    max_e = 0
    for e in range(args.epoch):
        s = env.reset()
        ep_reward = [0] * env.n_Veh
        outage_num = np.zeros((env.n_Veh, 1))
        up = np.zeros((env.n_Veh, 1))
        rews = [0] * env.n_Veh
        d = False
        j = 0
        while not d:
            actions = []
            for i in range(len(agents)):
                state = Batch(obs=[s[i]], state=None, info={})
                a = agents[i](state).act[0]
                actions.append(agents[i].exploration_noise(act=a.detach().numpy(), batch=state))
            actions = np.array(actions)
            s_, r, d, info = env.step(actions)

            for m in range(len(agents)):
                # with open("log.txt","a") as file:
                #     file.write("\n")
                #     file.write(f"{Batch(obs=s[m], act=actions[m], rew=r[m], done=d, obs_next=s_[m], info={})}")
                buffers[m].add(Batch(obs=s[m], act=actions[m], rew=r[m], done=d, obs_next=s_[m], info={}))
            if (buffers[0].__len__() >= args.batch_size) and (j % 5 == 0):
                args.exploration_noise *= 0.9999
                # actor_local_grads = []
                # critic1_local_grads = []
                # critic2_local_grads = []
                actor_global_grad = {}
                critic1_global_grad = {}
                critic2_global_grad = {}
                for n in range(len(agents)):
                    agents[n].set_exp_noise(GaussianNoise(sigma=args.exploration_noise))
                    # print(f"qian is:{n}")

                    # 联邦学习部分
                    # 每个agent在本地更新自己的模型参数
                    agents[n].update(sample_size=args.batch_size, buffer=buffers[n])
                    # print(f"after is:{n}")
                    # 计算梯度更新，即本地模型参数减去全局模型参数
                    actor_grad_update = {}
                    critic1_grad_update = {}
                    critic2_grad_update = {}
                    for key in agents[n].actor.state_dict().keys():
                        actor_grad_update[key] = (agents[n].actor.state_dict()[key] - global_agent.actor.state_dict()[key]) * w[n]
                        critic1_grad_update[key] = (agents[n].critic1.state_dict()[key] - global_agent.critic1.state_dict()[key]) * w[n]
                        critic2_grad_update[key] = (agents[n].critic2.state_dict()[key] - global_agent.critic2.state_dict()[key]) * w[n]
                    
                    # 服务器对梯度更新进行平均，并更新全局模型
                    if n == 0:
                        actor_global_grad = actor_grad_update
                        critic1_global_grad = critic1_grad_update
                        critic2_global_grad = critic2_grad_update
                    else:
                        for key in actor_grad_update.keys():
                            actor_global_grad[key] += actor_grad_update[key]
                        for key in critic1_grad_update.keys():
                            critic1_global_grad[key] += critic1_grad_update[key]
                        for key in critic2_grad_update.keys():
                            critic2_global_grad[key] += critic2_grad_update[key]


                    # actor_local_grads.append(actor_grad_update)
                    # critic1_local_grads.append(critic1_grad_update)
                    # critic2_local_grads.append(critic2_grad_update)

                    # 每个agent将自己的模型参数发送给服务器
                    # actor_params = agents[n].actor.state_dict()
                    # critic1_params = agents[n].critic1.state_dict()
                    # critic2_params = agents[n].critic2.state_dict()
                    # actor_old_params = agents[n].actor_old.state_dict()
                    # critic1_old_params = agents[n].critic1_old.state_dict()
                    # critic2_old_params = agents[n].critic2_old.state_dict()

                    # 服务器对所有agent的模型参数进行平均
                    # if n == 0:
                    #     global_actor_params = actor_params
                        # global_critic1_params = critic1_params
                        # global_critic2_params = critic2_params
                        # global_actor_old_params = actor_old_params
                        # global_critic1_old_params = critic1_old_params
                        # global_critic2_old_params = critic2_old_params
                    #     total_weight = 2 * w[n]
                    # else:
                    #     for key in actor_params.keys():
                    #         global_actor_params[key] += actor_params[key] * w[n]
                        # for key in critic1_params.keys():
                        #     global_critic1_params[key] += critic1_params[key] * w[n]
                        # for key in critic2_params.keys():
                        #     global_critic2_params[key] += critic2_params[key] * w[n]
                        # for key in actor_old_params.keys():
                        #     global_actor_old_params[key] = actor_old_params[key] * w[n]
                        # for key in critic1_old_params.keys():
                        #     global_critic1_old_params[key] = critic1_old_params[key] * w[n]
                        # for key in critic2_old_params.keys():
                        #     global_critic2_old_params[key] = critic2_old_params[key] * w[n]
                #         total_weight += 4 * w[n]
                # for key in actor_params.keys():
                #     global_actor_params[key] += actor_params[key] / total_weight

                    # print(f"global_actor_params[key] is:{global_actor_params[key]}")
                # for key in critic1_params.keys():
                #     global_critic1_params[key] += critic1_params[key] / total_weight

                #     # print(f"global_critic1_params[key] is:{global_critic1_params[key]}")
                # for key in critic2_params.keys():
                #     global_critic2_params[key] += critic2_params[key] / total_weight

                    # print(f"global_critic2_params[key] is:{global_critic2_params[key]}")
                # for key in actor_old_params.keys():
                #     global_actor_old_params[key] = actor_old_params[key] / total_weight
                # for key in critic1_old_params.keys():
                #     global_critic1_old_params[key] = critic1_old_params[key] / total_weight
                # for key in critic2_old_params.keys():
                #     global_critic2_old_params[key] = critic2_old_params[key] / total_weight


                # 服务器对梯度更新进行平均，并更新全局模型
                # actor_global_grad = average(actor_local_grads)
                # critic1_global_grad = average(critic1_local_grads)
                # critic2_global_grad = average(critic2_local_grads)

                # 更新全局模型参数，即加上平均后的梯度更新
                for key in global_agent.actor.state_dict().keys():
                    global_agent.actor.state_dict()[key] += actor_global_grad[key]
                for key in global_agent.critic1.state_dict().keys():
                    global_agent.critic1.state_dict()[key] += critic1_global_grad[key]
                for key in global_agent.critic2.state_dict().keys():
                    global_agent.critic2.state_dict()[key] += critic2_global_grad[key]

                # 服务器将平均后的模型参数发送给每个agent
                for n in range(len(agents)):
                    agents[n].actor.load_state_dict(global_agent.actor.state_dict())
                    agents[n].critic1.load_state_dict(global_agent.critic1.state_dict())
                    agents[n].critic2.load_state_dict(global_agent.critic2.state_dict())
                    # agents[n].actor_old.load_state_dict(global_actor_old_params)
                    # agents[n].critic1_old.load_state_dict(global_critic1_old_params)
                    # agents[n].critic2_old.load_state_dict(global_critic2_old_params)
            
            s = s_
            ep_reward += r
            outage_num += info['outage_time']
            rews += info['rews']
            up += info['up']
            j += 1
        harq_time = info['harq_time']
        with open("log.txt","a") as file:
            file.write("\n")
            file.write(f"trainning, Episode: {e}, Reward: {ep_reward}, Explore: {args.exploration_noise}, outage_num: {outage_num/harq_time}, throughput: {env.throughput/env.max_step}, up: {up/env.max_step}, dis:{env.vehicles[0].distance}, rews:{rews}, all rews is:{sum(env.throughput)/env.max_step}")
        with open('trainLTAT.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([env.throughput/10000])

        # 测试

        for i in range(agent_nums):
            agents[i] = agents[i].eval()

        s = test_envs.reset()
        ep_reward = [0] * env.n_Veh
        outage_num = np.zeros((env.n_Veh, 1))
        up = np.zeros((env.n_Veh, 1))
        rews = [0] * env.n_Veh
        d = False
        while not d:
            actions = []
            for i in range(len(agents)):
                state = Batch(obs=[s[i]], state=None, info={})
                a = agents[i](state).act[0]
                actions.append(a.detach().numpy())
            actions = np.array(actions)
            s_, r, d, info = test_envs.step(actions)
            s = s_
            ep_reward += r
            outage_num += info['outage_time']
            rews += info['rews']
            up += info['up']
        harq_time = info['harq_time']
        if sum(rews) > max_rew and (outage_num/harq_time)[0] < phi and (outage_num/harq_time)[1] < phi:
            max_rew = sum(rews)
            max_e = e
            max_th = sum(test_envs.throughput)/test_envs.max_step
        with open("log.txt","a") as file:
            file.write("\n")
            file.write(f"testing, Episode: {e}, Reward: {ep_reward}, Explore: {args.exploration_noise}, outage_num: {outage_num/harq_time}, throughput: {test_envs.throughput/test_envs.max_step}, up: {up}, dis:{env.vehicles[0].distance}, rews:{rews}, max_rew:{max_rew}, max_e:{max_e}, all th is:{sum(test_envs.throughput)/test_envs.max_step}")
        with open('testLTAT.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([env.throughput/10000])
    with open("log.txt","a") as file:
        file.write("\n")
        file.write(f"max_rew:{max_rew}, max_e:{max_e}, max_th:{max_th}")
