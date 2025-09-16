# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2022年05月15日
"""
import datetime
import pprint

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from Environment import Environ, IR_Environ
from model import ActorNet, CriticNet

import json
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.exploration import GaussianNoise
from tianshou.data import Batch, VectorReplayBuffer, ReplayBuffer, Collector, PrioritizedReplayBuffer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

"""
#####################  hyper parameters  ####################
"""
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use gpu or not')
parser.add_argument('--gpu_fraction', default=(0.5, 0), help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument('--task', type=str, default='Power-outage-train')
parser.add_argument('--reward-threshold', type=float, default=300000)
parser.add_argument('--seed', type=int, default=123, help='Value of random seed')
parser.add_argument('--buffer-size', type=int, default=20000)
parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128])
parser.add_argument('--actor-lr', type=float, default=1e-3)
parser.add_argument('--critic-lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--exploration-noise', type=float, default=0.7)
parser.add_argument("--start-timesteps", type=int, default=20000)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--step-per-epoch', type=int, default=1200)
parser.add_argument('--slot-per-test', type=int, default=6000)
parser.add_argument('--step-per-collect', type=int, default=1)
parser.add_argument('--update-per-step', type=float, default=1)
parser.add_argument('--episode-per-test', type=int, default=1)
parser.add_argument("--n-step", type=int, default=1)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--training-num', type=int, default=1)
parser.add_argument('--test-num', type=int, default=1)
parser.add_argument('--logdir', type=str, default='log')
parser.add_argument('--render', type=float, default=0.)
parser.add_argument(
    '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
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
args = parser.parse_args()


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
    最大重传次数为5
    """
    max_K = 5
    """
    IR-PER-DDPG-power训练
    """
    epsilon = [10**-1, 5 * 10**-2, 10**-2, 5 * 10**-3, 10**-3, 5 * 10**-4, 10**-4]
    beta = [5, 10, 15, 20, 25, 30, 35]
    for i in range(5, 40, 5):

        # Set random seed
        setup_seed(args.seed)

        env = IR_Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        train_envs = IR_Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        test_envs = IR_Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        args.max_action = env.action_space.high[0]
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        # model
        actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
        critic_net = CriticNet(args.state_shape, args.action_shape)
        actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

        # 创建策略
        agent = ts.policy.DDPGPolicy(actor=actor_net,
                                    actor_optim=actor_optim,
                                    critic=critic_net,
                                    critic_optim=critic_optim,
                                    gamma=args.gamma,
                                    tau=args.tau,
                                    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
                                    estimation_step=args.n_step,
                                    )

        # load a previous policy
        # if args.resume_path:
        #     agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        #     print("Loaded agent from: ", args.resume_path)

        # collector
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = PrioritizedReplayBuffer(args.buffer_size, 0.5, 0.5)
        train_collector = Collector(agent, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(agent, test_envs, exploration_noise=False)
        train_collector.collect(n_step=args.start_timesteps, random=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        args.algo_name = "ir-per_ddpg"
        log_name = os.path.join(args.task, args.algo_name, str(args.seed), str(max_K), "power"+str(i))
        log_path = os.path.join(args.logdir, log_name)

        # logger
        if args.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

        args.resume_path = os.path.join(log_path, 'policy.pth')

        # 保存该次训练的超参数
        with open(log_path + '/args.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # 训练
        def test_fn(epoch, env_step):
            agent.set_exp_noise(GaussianNoise(sigma=0))


        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold


        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


        if not args.watch:
            # trainer
            result = ts.trainer.offpolicy_trainer(
                agent,
                train_collector,
                test_collector,
                max_epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                step_per_collect=args.step_per_collect,
                update_per_step=args.update_per_step,
                episode_per_test=args.episode_per_test,
                batch_size=args.batch_size,
                test_fn=test_fn,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger
            )
            pprint.pprint(result)

    """
    xp-DDPG-power训练
    """
    for i in range(5, 40, 5):

        # Set random seed
        setup_seed(args.seed)

        env = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        train_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        test_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        args.max_action = env.action_space.high[0]
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        # model
        actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
        critic_net = CriticNet(args.state_shape, args.action_shape)
        actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

        # 创建策略
        agent = ts.policy.DDPGPolicy(actor=actor_net,
                                    actor_optim=actor_optim,
                                    critic=critic_net,
                                    critic_optim=critic_optim,
                                    gamma=args.gamma,
                                    tau=args.tau,
                                    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
                                    estimation_step=args.n_step,
                                    )

        # load a previous policy
        # if args.resume_path:
        #     agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        #     print("Loaded agent from: ", args.resume_path)

        # collector
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = ReplayBuffer(args.buffer_size)
        train_collector = Collector(agent, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(agent, test_envs, exploration_noise=False)
        train_collector.collect(n_step=args.start_timesteps, random=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        args.algo_name = "ddpg"
        log_name = os.path.join(args.task, args.algo_name, str(args.seed), str(max_K), "power"+str(i))
        log_path = os.path.join(args.logdir, log_name)

        # logger
        if args.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

        args.resume_path = os.path.join(log_path, 'policy.pth')

        # 保存该次训练的超参数
        with open(log_path + '/args.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # 训练
        def test_fn(epoch, env_step):
            agent.set_exp_noise(GaussianNoise(sigma=0))


        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold


        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


        if not args.watch:
            # trainer
            result = ts.trainer.offpolicy_trainer(
                agent,
                train_collector,
                test_collector,
                max_epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                step_per_collect=args.step_per_collect,
                update_per_step=args.update_per_step,
                episode_per_test=args.episode_per_test,
                batch_size=args.batch_size,
                test_fn=test_fn,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger,
            )
            pprint.pprint(result)

    """
    xp-PER-DDPG-power训练
    """
    for i in range(5, 40, 5):

        # Set random seed
        setup_seed(args.seed)

        env = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        train_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        test_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        args.max_action = env.action_space.high[0]
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        # model
        actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
        critic_net = CriticNet(args.state_shape, args.action_shape)
        actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

        # 创建策略
        agent = ts.policy.DDPGPolicy(actor=actor_net,
                                    actor_optim=actor_optim,
                                    critic=critic_net,
                                    critic_optim=critic_optim,
                                    gamma=args.gamma,
                                    tau=args.tau,
                                    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
                                    estimation_step=args.n_step,
                                    )

        # load a previous policy
        # if args.resume_path:
        #     agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        #     print("Loaded agent from: ", args.resume_path)

        # collector
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = PrioritizedReplayBuffer(args.buffer_size, 0.5, 0.5)
        train_collector = Collector(agent, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(agent, test_envs, exploration_noise=False)
        train_collector.collect(n_step=args.start_timesteps, random=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        args.algo_name = "per_ddpg"
        log_name = os.path.join(args.task, args.algo_name, str(args.seed), str(max_K), "power"+str(i))
        log_path = os.path.join(args.logdir, log_name)

        # logger
        if args.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

        args.resume_path = os.path.join(log_path, 'policy.pth')

        # 保存该次训练的超参数
        with open(log_path + '/args.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # 训练
        def test_fn(epoch, env_step):
            agent.set_exp_noise(GaussianNoise(sigma=0))


        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold


        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


        if not args.watch:
            # trainer
            result = ts.trainer.offpolicy_trainer(
                agent,
                train_collector,
                test_collector,
                max_epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                step_per_collect=args.step_per_collect,
                update_per_step=args.update_per_step,
                episode_per_test=args.episode_per_test,
                batch_size=args.batch_size,
                test_fn=test_fn,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger
            )
            pprint.pprint(result)

    """
    最大重传次数为3
    """
    max_K = 3
    """
    IR-PER-DDPG-power训练
    """
    for i in range(5, 40, 5):

        # Set random seed
        setup_seed(args.seed)

        env = IR_Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        train_envs = IR_Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        test_envs = IR_Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        args.max_action = env.action_space.high[0]
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        # model
        actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
        critic_net = CriticNet(args.state_shape, args.action_shape)
        actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

        # 创建策略
        agent = ts.policy.DDPGPolicy(actor=actor_net,
                                    actor_optim=actor_optim,
                                    critic=critic_net,
                                    critic_optim=critic_optim,
                                    gamma=args.gamma,
                                    tau=args.tau,
                                    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
                                    estimation_step=args.n_step,
                                    )

        # load a previous policy
        # if args.resume_path:
        #     agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        #     print("Loaded agent from: ", args.resume_path)

        # collector
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = PrioritizedReplayBuffer(args.buffer_size, 0.5, 0.5)
        train_collector = Collector(agent, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(agent, test_envs, exploration_noise=False)
        train_collector.collect(n_step=args.start_timesteps, random=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        args.algo_name = "ir-per_ddpg"
        log_name = os.path.join(args.task, args.algo_name, str(args.seed), str(max_K), "power"+str(i))
        log_path = os.path.join(args.logdir, log_name)

        # logger
        if args.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

        args.resume_path = os.path.join(log_path, 'policy.pth')

        # 保存该次训练的超参数
        with open(log_path + '/args.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # 训练
        def test_fn(epoch, env_step):
            agent.set_exp_noise(GaussianNoise(sigma=0))


        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold


        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


        if not args.watch:
            # trainer
            result = ts.trainer.offpolicy_trainer(
                agent,
                train_collector,
                test_collector,
                max_epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                step_per_collect=args.step_per_collect,
                update_per_step=args.update_per_step,
                episode_per_test=args.episode_per_test,
                batch_size=args.batch_size,
                test_fn=test_fn,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger
            )
            pprint.pprint(result)

    """
    xp-DDPG-power训练
    """
    for i in range(5, 40, 5):

        # Set random seed
        setup_seed(args.seed)

        env = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        train_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        test_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        args.max_action = env.action_space.high[0]
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        # model
        actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
        critic_net = CriticNet(args.state_shape, args.action_shape)
        actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

        # 创建策略
        agent = ts.policy.DDPGPolicy(actor=actor_net,
                                    actor_optim=actor_optim,
                                    critic=critic_net,
                                    critic_optim=critic_optim,
                                    gamma=args.gamma,
                                    tau=args.tau,
                                    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
                                    estimation_step=args.n_step,
                                    )

        # load a previous policy
        # if args.resume_path:
        #     agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        #     print("Loaded agent from: ", args.resume_path)

        # collector
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = ReplayBuffer(args.buffer_size)
        train_collector = Collector(agent, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(agent, test_envs, exploration_noise=False)
        train_collector.collect(n_step=args.start_timesteps, random=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        args.algo_name = "ddpg"
        log_name = os.path.join(args.task, args.algo_name, str(args.seed), str(max_K), "power"+str(i))
        log_path = os.path.join(args.logdir, log_name)

        # logger
        if args.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

        args.resume_path = os.path.join(log_path, 'policy.pth')

        # 保存该次训练的超参数
        with open(log_path + '/args.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # 训练
        def test_fn(epoch, env_step):
            agent.set_exp_noise(GaussianNoise(sigma=0))


        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold


        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


        if not args.watch:
            # trainer
            result = ts.trainer.offpolicy_trainer(
                agent,
                train_collector,
                test_collector,
                max_epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                step_per_collect=args.step_per_collect,
                update_per_step=args.update_per_step,
                episode_per_test=args.episode_per_test,
                batch_size=args.batch_size,
                test_fn=test_fn,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger,
            )
            pprint.pprint(result)

    """
    xp-PER-DDPG-power训练
    """
    for i in range(5, 40, 5):

        # Set random seed
        setup_seed(args.seed)

        env = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        train_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        test_envs = Environ(i, max_K, epsilon=epsilon[int(i/5)-1], training=True)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        args.max_action = env.action_space.high[0]
        print("Observations shape:", args.state_shape)
        print("Actions shape:", args.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

        # model
        actor_net = ActorNet(args.state_shape, args.action_shape, action_bound=1)
        critic_net = CriticNet(args.state_shape, args.action_shape)
        actor_optim = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

        # 创建策略
        agent = ts.policy.DDPGPolicy(actor=actor_net,
                                    actor_optim=actor_optim,
                                    critic=critic_net,
                                    critic_optim=critic_optim,
                                    gamma=args.gamma,
                                    tau=args.tau,
                                    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
                                    estimation_step=args.n_step,
                                    )

        # load a previous policy
        # if args.resume_path:
        #     agent.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        #     print("Loaded agent from: ", args.resume_path)

        # collector
        if args.training_num > 1:
            buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        else:
            buffer = PrioritizedReplayBuffer(args.buffer_size, 0.5, 0.5)
        train_collector = Collector(agent, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(agent, test_envs, exploration_noise=False)
        train_collector.collect(n_step=args.start_timesteps, random=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        args.algo_name = "per_ddpg"
        log_name = os.path.join(args.task, args.algo_name, str(args.seed), str(max_K), "power"+str(i))
        log_path = os.path.join(args.logdir, log_name)

        # logger
        if args.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

        args.resume_path = os.path.join(log_path, 'policy.pth')

        # 保存该次训练的超参数
        with open(log_path + '/args.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # 训练
        def test_fn(epoch, env_step):
            agent.set_exp_noise(GaussianNoise(sigma=0))


        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold


        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


        if not args.watch:
            # trainer
            result = ts.trainer.offpolicy_trainer(
                agent,
                train_collector,
                test_collector,
                max_epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                step_per_collect=args.step_per_collect,
                update_per_step=args.update_per_step,
                episode_per_test=args.episode_per_test,
                batch_size=args.batch_size,
                test_fn=test_fn,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger
            )
            pprint.pprint(result)
