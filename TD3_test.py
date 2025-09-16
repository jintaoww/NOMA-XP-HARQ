# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2022年05月09日
"""
from ast import arg
import datetime
import pprint
import re
from tkinter import font

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from Environment import Environ, IR_Environ
from model import ActorNet, CriticNet

import json
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.exploration import GaussianNoise
from tianshou.data import Batch, VectorReplayBuffer, ReplayBuffer, Collector
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic


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
    # 控制GPU资源使用的两种方法
    # （1）直接限制gpu的使用率
    print(torch.cuda.is_available())
    torch.cuda.set_per_process_memory_fraction(0.5, 0)

    """
    最大传输次数为5
    """
    max_k = 5
    # """
    # IR-PER-DDPG-power测试
    # """
    # IR_PER_DDPG_outage_all_5 = []
    # IR_PER_DDPG_throughput_all_5 = []
    # for i in range(5, 40, 5):

    #     ir_per_ddpg_args_path = "D:\PythonProject\constrain-HARQ_Throughout_rate_selection\DDPG_XP_HARQ_Throughout_Optimization\log\Power-outage-train\ir-per_ddpg/123/" + str(max_k) + "\power" + str(i) + "/args.json"

    #     args = args_read(args_path=ir_per_ddpg_args_path)

    #     # Set random seed
    #     setup_seed(args.seed)

    #     env = IR_Environ(i, max_k)
    #     train_envs = IR_Environ(i, max_k)
    #     test_envs = IR_Environ(i, max_k)
        

    #     args.state_shape = env.observation_space.shape or env.observation_space.n
    #     args.action_shape = env.action_space.shape or env.action_space.n
    #     args.max_action = env.action_space.high[0]
    #     print("Observations shape:", args.state_shape)
    #     print("Actions shape:", args.action_shape)
    #     print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    #     # model
    #     actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
    #     critic_net = CriticNet(args.state_shape, args.action_shape)
    #     actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    #     critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

    #     # 创建策略
    #     agent = ts.policy.DDPGPolicy(actor=actor_net,
    #                                 actor_optim=actor_optim,
    #                                 critic=critic_net,
    #                                 critic_optim=critic_optim,
    #                                 gamma=args.gamma,
    #                                 tau=args.tau,
    #                                 exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    #                                 estimation_step=args.n_step,
    #                                 )

    #     # load a previous policy
    #     if args.resume_path:
    #         agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #         print("Loaded agent from: ", args.resume_path)

    #     r_episode = np.zeros(args.epoch)
    #     outage_num = 0
    #     harq_time = 0
    #     for i in range(args.epoch):
    #         s = env.reset()
    #         d = False
    #         reward_sum = 0

    #         while not d:
    #             state = Batch(obs=[s], state=None, info={})
    #             a = agent(state).act[0]
    #             s_, r, d, info = env.step(a.detach().numpy())
    #             outage_num += info["outage_time"]
    #             reward_sum += r
    #             # env.render()

    #             s = s_
    #         harq_time += info['harq_time']
    #         r_episode[i] = reward_sum / env.step_time
    #     print('Average rewards using IR-PER-DDPG:', np.mean(r_episode), f"\nAverage outage probability using IR-PER-DDPG: {outage_num/harq_time}")
    #     IR_PER_DDPG_throughput_all_5.append(np.mean(r_episode))
    #     IR_PER_DDPG_outage_all_5.append(outage_num/harq_time)
        # r_episode_mean_PER_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
        # plt.figure(i)
        # plt.plot(100 * np.arange(len(r_episode_mean_PER_DDPG)), r_episode_mean_PER_DDPG)
        # plt.xlabel(u'时隙(TS)')
        # plt.ylabel(u'HARQ系统吞吐量(bit/s)')
        # plt.title(u'基于PER-DDPG的资源分配策略')
    
    # """
    # XP-DDPG-power测试
    # """
    # XP_DDPG_outage_all_5 = []
    # XP_DDPG_throughput_all_5 = []
    # for i in range(5, 40, 5):
    #     per_ddpg_args_path = "D:\PythonProject\constrain-HARQ_Throughout_rate_selection\DDPG_XP_HARQ_Throughout_Optimization\log\Power-outage-train\ddpg/123/" + str(max_k) + "\power" + str(i) + "/args.json"

    #     args = args_read(args_path=per_ddpg_args_path)

    #     # Set random seed
    #     setup_seed(args.seed)

    #     env = Environ(i, max_k)
    #     train_envs = Environ(i, max_k)
    #     test_envs = Environ(i, max_k)

    #     args.state_shape = env.observation_space.shape or env.observation_space.n
    #     args.action_shape = env.action_space.shape or env.action_space.n
    #     args.max_action = env.action_space.high[0]
    #     print("Observations shape:", args.state_shape)
    #     print("Actions shape:", args.action_shape)
    #     print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    #     # model
    #     actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
    #     critic_net = CriticNet(args.state_shape, args.action_shape)
    #     actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    #     critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

    #     # 创建策略
    #     agent = ts.policy.DDPGPolicy(actor=actor_net,
    #                                 actor_optim=actor_optim,
    #                                 critic=critic_net,
    #                                 critic_optim=critic_optim,
    #                                 gamma=args.gamma,
    #                                 tau=args.tau,
    #                                 exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    #                                 estimation_step=args.n_step,
    #                                 )

    #     # load a previous policy
    #     if args.resume_path:
    #         agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #         print("Loaded agent from: ", args.resume_path)

    #     r_episode = np.zeros((args.epoch, 60))
    #     outage_num = 0
    #     harq_time = 0
    #     for i in range(args.epoch):
    #         s = env.reset()
    #         d = False
    #         reward_sum = 0

    #         while not d:
    #             state = Batch(obs=[s], state=None, info={})
    #             a = agent(state).act[0]
    #             s_, r, d, info = env.step(a.detach().numpy())
    #             outage_num += info['outage_time']
    #             # env.render()
    #             reward_sum += r

    #             s = s_
    #         harq_time += info['harq_time']
    #         r_episode[i] = reward_sum / env.step_time

    #     print('Average rewards using XP-DDPG:', np.mean(r_episode), f"\nAverage outage probability using XP-DDPG: {outage_num/harq_time}")
    #     XP_DDPG_throughput_all_5.append(np.mean(r_episode))
    #     XP_DDPG_outage_all_5.append(outage_num/harq_time)
    #     # r_episode_mean_PER_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
    #     # plt.figure(i)
    #     # plt.plot(100 * np.arange(len(r_episode_mean_PER_DDPG)), r_episode_mean_PER_DDPG)
    #     # plt.xlabel(u'时隙(TS)')
    #     # plt.ylabel(u'HARQ系统吞吐量(bit/s)')
    #     # plt.title(u'基于PER-DDPG的资源分配策略')

    """
    XP-PER-DDPG-power测试
    """
    epsilon = [10 ** -3] * 7
    XP_PER_DDPG_outage_all_5 = []
    XP_PER_DDPG_throughput_all_5 = []
    beta = [.5, 1, 1.8, 1.9, 2.1, 2.2, 2.3]

    for i in range(5, 40, 5):
        per_ddpg_args_path = "D:\PythonProject\constrain-HARQ_Throughout_rate_selection-test\DDPG_XP_HARQ_Throughout_Optimization\log\TD3-Power-outage-train\per_ddpg/123/" + str(max_k) + "\power" + str(i) + "/args.json"

        args = args_read(args_path=per_ddpg_args_path)

        # Set random seed
        setup_seed(args.seed)

        env = Environ(i, max_k, seed=args.seed)
        train_envs = Environ(i, max_k, seed=args.seed)
        test_envs = Environ(i, max_k, seed=args.seed)
        # test_envs = Environ(i, max_k, epsilon=10 ** -3, beta=beta[int(i/5)-1], training=True, seed=args.seed)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        args.max_action = env.action_space.high[0]
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        # model
        actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=args.max_action)
        actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
        critic1_net = CriticNet(args.state_shape, args.action_shape)
        critic1_optim = torch.optim.Adam(critic1_net.parameters(), lr=args.critic_lr)
        critic2_net = CriticNet(args.state_shape, args.action_shape)
        critic2_optim = torch.optim.Adam(critic2_net.parameters(), lr=args.critic_lr)

        # 创建策略
        agent = ts.policy.TD3Policy(actor_net,
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

        # load a previous policy
        if args.resume_path:
            agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
            print("Loaded agent from: ", args.resume_path)

        # 测试基于PER-DDPG的资源分配策略
        r_episode = np.zeros((1, 1))
        outage_num = 0
        harq_time = 0
        for i in range(1):
            s = test_envs.reset()
            d = False
            reward_sum = 0

            while not d:
                state = Batch(obs=[s], state=None, info={})
                a = agent(state).act[0]
                s_, r, d, info = test_envs.step(a.detach().numpy())
                outage_num += info['outage_time']
                # env.render()
                reward_sum += r

                s = s_
            harq_time += info['harq_time']
            # print(f"reward_sum:{reward_sum}, step_time:{test_envs.step_time}")
            r_episode[i] = reward_sum / test_envs.step_time


        print('Average rewards using XP-PER-DDPG:', np.mean(r_episode), f"\nAverage outage probability using XP-PER-DDPG: {outage_num/harq_time}")
        XP_PER_DDPG_throughput_all_5.append(np.mean(r_episode))
        XP_PER_DDPG_outage_all_5.append(outage_num/harq_time)
        # r_episode_mean_PER_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
        # plt.figure(i)
        # plt.plot(100 * np.arange(len(r_episode_mean_PER_DDPG)), r_episode_mean_PER_DDPG)
        # plt.xlabel(u'时隙(TS)')
        # plt.ylabel(u'HARQ系统吞吐量(bit/s)')
        # plt.title(u'基于PER-DDPG的资源分配策略')


    # """
    # 最大传输为3
    # """
    # max_k = 3
    # """
    # IR-PER-DDPG-power测试
    # """
    # IR_PER_DDPG_outage_all_3 = []
    # IR_PER_DDPG_throughput_all_3 = []
    # for i in range(5, 40, 5):

    #     ir_per_ddpg_args_path = "D:\PythonProject\constrain-HARQ_Throughout_rate_selection\DDPG_XP_HARQ_Throughout_Optimization\log\Power-outage-train\ir-per_ddpg/123/" + str(max_k) + "\power" + str(i) + "/args.json"

    #     args = args_read(args_path=ir_per_ddpg_args_path)

    #     # Set random seed
    #     setup_seed(args.seed)

    #     env = IR_Environ(i, max_k)
    #     train_envs = IR_Environ(i, max_k)
    #     test_envs = IR_Environ(i, max_k)

    #     args.state_shape = env.observation_space.shape or env.observation_space.n
    #     args.action_shape = env.action_space.shape or env.action_space.n
    #     args.max_action = env.action_space.high[0]
    #     print("Observations shape:", args.state_shape)
    #     print("Actions shape:", args.action_shape)
    #     print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    #     # model
    #     actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
    #     critic_net = CriticNet(args.state_shape, args.action_shape)
    #     actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    #     critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

    #     # 创建策略
    #     agent = ts.policy.DDPGPolicy(actor=actor_net,
    #                                 actor_optim=actor_optim,
    #                                 critic=critic_net,
    #                                 critic_optim=critic_optim,
    #                                 gamma=args.gamma,
    #                                 tau=args.tau,
    #                                 exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    #                                 estimation_step=args.n_step,
    #                                 )

    #     # load a previous policy
    #     if args.resume_path:
    #         agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #         print("Loaded agent from: ", args.resume_path)

    #     r_episode = np.zeros(args.epoch)
    #     outage_num = 0
    #     harq_time = 0
    #     for i in range(args.epoch):
    #         s = env.reset()
    #         d = False
    #         reward_sum = 0

    #         while not d:
    #             state = Batch(obs=[s], state=None, info={})
    #             a = agent(state).act[0]
    #             s_, r, d, info = env.step(a.detach().numpy())
    #             outage_num += info["outage_time"]
    #             reward_sum += r
    #             # env.render()

    #             s = s_
    #         harq_time += info['harq_time']
    #         r_episode[i] = reward_sum / env.step_time
    #     print('Average rewards using IR-PER-DDPG:', np.mean(r_episode), f"\nAverage outage probability using IR-PER-DDPG: {outage_num/harq_time}")
    #     IR_PER_DDPG_throughput_all_3.append(np.mean(r_episode))
    #     IR_PER_DDPG_outage_all_3.append(outage_num/harq_time)
    #     # r_episode_mean_PER_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
    #     # plt.figure(i)
    #     # plt.plot(100 * np.arange(len(r_episode_mean_PER_DDPG)), r_episode_mean_PER_DDPG)
    #     # plt.xlabel(u'时隙(TS)')
    #     # plt.ylabel(u'HARQ系统吞吐量(bit/s)')
    #     # plt.title(u'基于PER-DDPG的资源分配策略')
    
    # """
    # XP-DDPG-power测试
    # """
    # XP_DDPG_outage_all_3 = []
    # XP_DDPG_throughput_all_3 = []
    # for i in range(5, 40, 5):

    #     per_ddpg_args_path = "D:\PythonProject\constrain-HARQ_Throughout_rate_selection\DDPG_XP_HARQ_Throughout_Optimization\log\Power-outage-train\ddpg/123/" + str(max_k) + "\power" + str(i) + "/args.json"

    #     args = args_read(args_path=per_ddpg_args_path)

    #     # Set random seed
    #     setup_seed(args.seed)

    #     env = Environ(i, max_k)
    #     train_envs = Environ(i, max_k)
    #     test_envs = Environ(i, max_k)

    #     args.state_shape = env.observation_space.shape or env.observation_space.n
    #     args.action_shape = env.action_space.shape or env.action_space.n
    #     args.max_action = env.action_space.high[0]
    #     print("Observations shape:", args.state_shape)
    #     print("Actions shape:", args.action_shape)
    #     print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    #     # model
    #     actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
    #     critic_net = CriticNet(args.state_shape, args.action_shape)
    #     actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    #     critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

    #     # 创建策略
    #     agent = ts.policy.DDPGPolicy(actor=actor_net,
    #                                 actor_optim=actor_optim,
    #                                 critic=critic_net,
    #                                 critic_optim=critic_optim,
    #                                 gamma=args.gamma,
    #                                 tau=args.tau,
    #                                 exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    #                                 estimation_step=args.n_step,
    #                                 )

    #     # load a previous policy
    #     if args.resume_path:
    #         agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #         print("Loaded agent from: ", args.resume_path)

    #     r_episode = np.zeros((args.epoch, 60))
    #     outage_num = 0
    #     harq_time = 0
    #     for i in range(args.epoch):
    #         s = env.reset()
    #         d = False
    #         reward_sum = 0

    #         while not d:
    #             state = Batch(obs=[s], state=None, info={})
    #             a = agent(state).act[0]
    #             s_, r, d, info = env.step(a.detach().numpy())
    #             outage_num += info['outage_time']
    #             # env.render()
    #             reward_sum += r

    #             s = s_
    #         harq_time += info['harq_time']
    #         r_episode[i] = reward_sum / env.step_time

    #     print('Average rewards using XP-DDPG:', np.mean(r_episode), f"\nAverage outage probability using XP-DDPG: {outage_num/harq_time}")
    #     XP_DDPG_throughput_all_3.append(np.mean(r_episode))
    #     XP_DDPG_outage_all_3.append(outage_num/harq_time)
    #     # r_episode_mean_PER_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
    #     # plt.figure(i)
    #     # plt.plot(100 * np.arange(len(r_episode_mean_PER_DDPG)), r_episode_mean_PER_DDPG)
    #     # plt.xlabel(u'时隙(TS)')
    #     # plt.ylabel(u'HARQ系统吞吐量(bit/s)')
    #     # plt.title(u'基于PER-DDPG的资源分配策略')

    # """
    # XP-PER-DDPG-power测试
    # """
    # XP_PER_DDPG_outage_all_3 = []
    # XP_PER_DDPG_throughput_all_3 = []
    # for i in range(5, 40, 5):

    #     per_ddpg_args_path = "D:\PythonProject\constrain-HARQ_Throughout_rate_selection\DDPG_XP_HARQ_Throughout_Optimization\log\Power-outage-train\per_ddpg/123/" + str(max_k) + "\power" + str(i) + "/args.json"

    #     args = args_read(args_path=per_ddpg_args_path)

    #     # Set random seed
    #     setup_seed(args.seed)

    #     env = Environ(i, max_k)
    #     train_envs = Environ(i, max_k)
    #     test_envs = Environ(i, max_k)

    #     args.state_shape = env.observation_space.shape or env.observation_space.n
    #     args.action_shape = env.action_space.shape or env.action_space.n
    #     args.max_action = env.action_space.high[0]
    #     print("Observations shape:", args.state_shape)
    #     print("Actions shape:", args.action_shape)
    #     print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    #     # model
    #     actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
    #     critic_net = CriticNet(args.state_shape, args.action_shape)
    #     actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    #     critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

    #     # 创建策略
    #     agent = ts.policy.DDPGPolicy(actor=actor_net,
    #                                 actor_optim=actor_optim,
    #                                 critic=critic_net,
    #                                 critic_optim=critic_optim,
    #                                 gamma=args.gamma,
    #                                 tau=args.tau,
    #                                 exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    #                                 estimation_step=args.n_step,
    #                                 )

    #     # load a previous policy
    #     if args.resume_path:
    #         agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #         print("Loaded agent from: ", args.resume_path)

    #     # 测试基于PER-DDPG的资源分配策略
    #     r_episode = np.zeros((args.epoch, 60))
    #     outage_num = 0
    #     harq_time = 0
    #     for i in range(args.epoch):
    #         s = env.reset()
    #         d = False
    #         reward_sum = 0

    #         while not d:
    #             state = Batch(obs=[s], state=None, info={})
    #             a = agent(state).act[0]
    #             s_, r, d, info = env.step(a.detach().numpy())
    #             outage_num += info['outage_time']
    #             # env.render()
    #             reward_sum += r

    #             s = s_
    #         harq_time += info['harq_time']
    #         r_episode[i] = reward_sum / env.step_time


    #     print('Average rewards using XP-PER-DDPG:', np.mean(r_episode), f"\nAverage outage probability using XP-PER-DDPG: {outage_num/harq_time}")
    #     XP_PER_DDPG_throughput_all_3.append(np.mean(r_episode))
    #     XP_PER_DDPG_outage_all_3.append(outage_num/harq_time)
    #     # r_episode_mean_PER_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
    #     # plt.figure(i)
    #     # plt.plot(100 * np.arange(len(r_episode_mean_PER_DDPG)), r_episode_mean_PER_DDPG)
    #     # plt.xlabel(u'时隙(TS)')
    #     # plt.ylabel(u'HARQ系统吞吐量(bit/s)')
    #     # plt.title(u'基于PER-DDPG的资源分配策略')

    # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
    x = np.arange(5, 40, 5)
    plt.figure(1)
    # plt.semilogy(x, IR_PER_DDPG_outage_all_3, marker='o')
    # plt.semilogy(x, XP_DDPG_outage_all_3, marker='>')
    # plt.semilogy(x, XP_PER_DDPG_outage_all_3, marker='*')
    # plt.semilogy(x, IR_PER_DDPG_outage_all_5, linestyle='--', marker='o')
    # plt.semilogy(x, XP_DDPG_outage_all_5, linestyle='--', marker='>')
    plt.semilogy(x, XP_PER_DDPG_outage_all_5, linestyle='--', marker='*')
    plt.semilogy(x, epsilon, linestyle='--', marker='>')
    plt.xlabel(u'power(dB)', fontsize=15)
    plt.ylabel(u'HARQ system outage probability (%)', fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend([u"XP-PER-DDPG:K=5"], fontsize=10)
    plt.figure(1).savefig('./Documentation/TD3_Outages.eps', dpi=600, format='eps')

    plt.figure(2)
    # plt.plot(x, IR_PER_DDPG_throughput_all_3, marker='o')
    # plt.plot(x, XP_DDPG_throughput_all_3, marker='>')
    # plt.plot(x, XP_PER_DDPG_throughput_all_3, marker='*')
    # plt.plot(x, IR_PER_DDPG_throughput_all_5, linestyle='--', marker='o')
    # plt.plot(x, XP_DDPG_throughput_all_5, linestyle='--', marker='>')

    plt.plot(x, XP_PER_DDPG_throughput_all_5, linestyle='--', marker='*')
    plt.xlabel(u'power(DB)', fontsize=15)
    plt.ylabel(u'HARQ system throughput(bit/s)', fontsize=15)
    plt.legend([u"XP-PER-DDPG:K=5"], fontsize=10)
    plt.figure(2).savefig('./Documentation/TD3_Throughput_Power_All.eps', dpi=600, format='eps')

    plt.show()