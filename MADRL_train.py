"""
作者：Admin
日期：2022年05月09日
"""
from ast import arg
import datetime
import pprint
import re
from tkinter import font
import copy

import csv

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# from test_xp_harq import NOMA_Environ
from MADRL_test_noma_harq_env import NOMA_Environ
from model import ActorNet, CriticNet

import json
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.exploration import GaussianNoise
from tianshou.data import Batch, VectorReplayBuffer, ReplayBuffer, Collector, PrioritizedReplayBuffer
from tianshou.utils import TensorboardLogger
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
parser.add_argument('--actor-lr', type=float, default=3e-4)
parser.add_argument('--critic-lr', type=float, default=3e-4)
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
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)

    """
    XP-PER-DDPG-power测试
    """

    phi = 10**-2
    args = parser.parse_args()

    # Set random seed
    setup_seed(args.seed)

    env = NOMA_Environ(10, 1, args.seed, phi=phi, beta=[15.6, 15.6], training=True)
    train_envs = NOMA_Environ(10, 1, args.seed, training=True)
    test_envs = NOMA_Environ(10, 1, args.seed, training=False)
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
    critic1_net = CriticNet(args.state_shape, args.action_shape)
    critic1_optim = torch.optim.Adam(critic1_net.parameters(), lr=args.critic_lr)
    critic2_net = CriticNet(args.state_shape, args.action_shape)
    critic2_optim = torch.optim.Adam(critic2_net.parameters(), lr=args.critic_lr)

    # model
    # actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
    # actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    # critic1_net = CriticNet(args.state_shape, args.action_shape)
    # critic1_optim = torch.optim.Adam(critic1_net.parameters(), lr=args.critic_lr)
    # critic2_net = CriticNet(args.state_shape, args.action_shape)
    # critic2_optim = torch.optim.Adam(critic2_net.parameters(), lr=args.critic_lr)

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

    # load a previous policy
    # if args.resume_path:
    #     agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #     print("Loaded agent from: ", args.resume_path)


    buffer = PrioritizedReplayBuffer(args.buffer_size, 0.5, 0.5)
    # buffer = ReplayBuffer(args.buffer_size)
    for i in range(agent_nums):
        agents.append(copy.deepcopy(agent))
        agents[i] = agents[i].train()
        buffers.append(copy.deepcopy(buffer))
    
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
            # print(s)
            actions = []
            for i in range(len(agents)):
                state = Batch(obs=[s[i]], state=None, info={})
                # print(state)
                a = agents[i](state).act[0]
                # print(agents[i].exploration_noise(act=a.detach().numpy(), batch=state))
                actions.append(agents[i].exploration_noise(act=a.detach().numpy(), batch=state))
            actions = np.array(actions)
            s_, r, d, info = env.step(actions)
            for m in range(len(agents)):
                # print(Batch(obs=s[i], act=actions[i], rew=r[i], done=d[i], obs_next=s_[i], info={}))
                buffers[m].add(Batch(obs=s[m], act=actions[m], rew=r[m], done=d, obs_next=s_[m], info={}))
            # print('Episode:', e, j,buffers[0].__len__())
            if (buffers[0].__len__() >= args.batch_size) and (j % 5 == 0):
                args.exploration_noise *= 0.9999
                for n in range(len(agents)):
                    agents[n].set_exp_noise(GaussianNoise(sigma=args.exploration_noise))
                    agents[n].update(sample_size=args.batch_size, buffer=buffers[n])
            s = s_
            ep_reward += r
            outage_num += info['outage_time']
            rews += info['rews']
            up += info['up']
            j += 1
        harq_time = info['harq_time']
        with open("log.txt","a") as file:
            file.write("\n")
            file.write(f"trainning, Episode: {e}, Reward: {ep_reward}, Explore: {args.exploration_noise}, outage_num: {outage_num/harq_time}, throughput: {env.throughput/10000}, up: {up/10000}, dis:{env.vehicles[0].distance}, rews:{rews}, all rews is:{sum(env.throughput)/10000}")
        with open('trainLTAT.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([env.throughput/10000])
            # file.write("And more…")
        print('trainning, Episode:', e, f' Reward: {ep_reward}', f"Explore: {args.exploration_noise}, outage_num: {outage_num/harq_time}, throughput: {env.throughput/10000}, up: {up/10000}, dis:{env.vehicles[0].distance}, rews:{rews}, all rews is:{sum(env.throughput)/10000}")
        
        s = test_envs.reset()
        ep_reward = [0] * env.n_Veh
        outage_num = np.zeros((env.n_Veh, 1))
        up = np.zeros((env.n_Veh, 1))
        rews = [0] * env.n_Veh
        d = False
        while not d:
            # print(s)
            actions = []
            for i in range(len(agents)):
                state = Batch(obs=[s[i]], state=None, info={})
                # print(state)
                a = agents[i](state).act[0]
                # print(agents[i].exploration_noise(act=a.detach().numpy(), batch=state))
                actions.append(a.detach().numpy())
            actions = np.array(actions)
            s_, r, d, info = test_envs.step(actions)
            # print('Episode:', e, j,buffers[0].__len__())
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

    print(f"max_rew:{max_rew}, max_e:{max_e}")
        # print('Episode:', e, f' Reward: {ep_reward}', f"Explore: {args.exploration_noise}, outage_num: {outage_num/harq_time}, dis:{env.vehicles[0].distance}")




    # # log
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # args.algo_name = "per_ddpg"
    # log_name = os.path.join(args.task, args.algo_name, str(args.seed), str(max_K), "power"+str(i))
    # log_path = os.path.join(args.logdir, log_name)

    # # logger
    # if args.logger == "wandb":
    #     logger = WandbLogger(
    #         save_interval=1,
    #         name=log_name.replace(os.path.sep, "__"),
    #         run_id=args.resume_id,
    #         config=args,
    #         project=args.wandb_project,
    #     )
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # if args.logger == "tensorboard":
    #     logger = TensorboardLogger(writer)
    # else:  # wandb
    #     logger.load(writer)

    # args.resume_path = os.path.join(log_path, 'policy.pth')

    # # 保存该次训练的超参数
    # with open(log_path + '/args.json', 'wt') as f:
    #     json.dump(vars(args), f, indent=4)

    # # 训练
    # def test_fn(epoch, env_step):
    #     agent.set_exp_noise(GaussianNoise(sigma=0))


    # def stop_fn(mean_rewards):
    #     return mean_rewards >= args.reward_threshold


    # def save_best_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


    # if not args.watch:
    #     # trainer
    #     result = ts.trainer.offpolicy_trainer(
    #         agent,
    #         train_collector,
    #         test_collector,
    #         max_epoch=args.epoch,
    #         step_per_epoch=args.step_per_epoch,
    #         step_per_collect=args.step_per_collect,
    #         update_per_step=args.update_per_step,
    #         episode_per_test=args.episode_per_test,
    #         batch_size=args.batch_size,
    #         stop_fn=stop_fn,
    #         save_best_fn=save_best_fn,

    #         logger=logger
    #     )
    #     pprint.pprint(result)