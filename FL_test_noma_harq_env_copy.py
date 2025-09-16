# -*- coding:utf-8 -*-
"""
作者：Admin
"""
from typing import Optional, Union

import gym
import numpy as np
from gym import spaces
from gym.core import ObsType
from pettingzoo.utils import agent_selector

def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c

class V2IChannels:
    """Simulator of the V2I channels (V2I信道模拟器)"""

    def __init__(self, n_veh, n_rb):
        self.n_Veh = n_veh
        self.n_RB = n_rb
        self.distances = None
        self.PathLoss = None
        self.FastFading = None
        self.h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_Veh, 1)) +
                            1j * np.random.normal(size=(self.n_Veh, 1)))
        self.rho = .4

    def update_distances(self, distances):
        self.distances = distances

    def update_path_loss(self):
        # 用于更新V2I之间的路径损失PL
        self.PathLoss = np.zeros(len(self.distances))
        for i in range(len(self.distances)):
            # self.PathLoss[i] = -(122.0 + 38.0 * np.log10(self.distances[i] / 1000))
            # self.PathLoss[i] = 10. ** (self.PathLoss[i] / 10)
            self.PathLoss[i] = self.distances[i] ** (-2)

    """
        初始化快（小规模）衰落的信道特征h(dB)，self.FastFading为一个二维数组，
        第二维是资源块的数量，即V2I link的每个子频带，都有着它自己的信道特性h(self.FastFading)
        二维数组的每一个元素都是实部和虚部都服从正太分布的一个复信特性
    """
    def update_fast_fading(self):
        # h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_Veh, self.n_RB )) +
        #                     1j * np.random.normal(size=(self.n_Veh, self.n_RB)))
        # self.FastFading = 20 * np.log10(np.abs(h))
        self.h = self.rho * self.h + np.sqrt(1 - np.power(self.rho, 2)) * np.random.normal(size=(self.n_Veh, 1))
        self.FastFading = np.abs(self.h)


class Vehicle:
    """
        Vehicle simulator: include all the information for a vehicle
        车辆类，每一辆车的信息包括：位置、方向、速度、它到的3个neighbors的距离、目的地
    """
    def __init__(self, start_distance, velocity):
        self.distance = start_distance  # 每个位置信号包含着（x, y）坐标的两个变量
        self.velocity = velocity  # 0m/s


class NOMA_Environ(gym.Env):
    """
        环境模拟器：（1）为agent提供状态s和反馈reward
        （2）根据agent所采取的动作action，环境会返回更新的状态s（t+1）
    """

    # 初始化环境参数
    def __init__(self, power_db=20, max_K=3, seed=123, phi=10**-3, epsilon=0.2, beta=[50, 50], alpha=[1.5, 1.5], training=False):
        np.random.seed(seed)
        self.se = seed
        self.time_step = 0.1  # 更新车辆位置信息的时间间隔
        self.vehicles = []  # 用于存储摆放在环境中的车辆对象（有Vehicle类生成）
        self.PO = 10.**(power_db/10) * max_K
        self.sig2_dB = -30  # 环境高斯噪声功率（dB）
        self.sig2 = 10**(self.sig2_dB/10)  # 环境高斯噪声功率（W）
        self.delta_distance = []
        self.n_RB = 1  # 子频带的数量
        self.n_Veh = 2  # 车辆的数量
        self.state_dim = self.n_Veh * self.n_RB
        self.action_dim = self.n_Veh * self.n_RB
        self.action_bound = 1.0
        self.V2IChannels = V2IChannels(self.n_Veh, self.n_RB)  # 创建V2I信道对象
        self.reset_pointer = 0
        self.radius = 1000
        self.start_velocity = 0  # 用户的初始速度，此处为0m/s
        self.V2I_channels_abs = None
        self.V2I_channels_with_fast_fading = None
        self.Maximum_number_of_transmission_K = max_K

        self.lambd = np.array([0.15]*self.n_Veh)
        self.mu = np.array([0.1]*self.n_Veh)
        self.training = training
        self.beta = beta
        self.alpha = alpha
        # self.beta = np.array([beta]*self.n_Veh)
        self.phi = np.array([phi]*self.n_Veh)
        self.epsilon = epsilon
        self.outage_num = np.zeros((self.n_Veh, 1))
        self.harq_time = np.zeros((self.n_Veh, 1))
        self.avg_trans = np.array([self.Maximum_number_of_transmission_K]*self.n_Veh)

        self.throughput = np.zeros(self.n_Veh)
        self.out_throughput = np.zeros(self.n_Veh)
        self.accumulation_power = np.zeros((self.n_Veh, 1))
        self.mutual_information = np.zeros((self.n_Veh, 1))
        self.accumulation_rate = np.zeros((self.n_Veh, 1))
        self.current_transmission_time = np.ones((self.n_Veh, 1))
        

        self.agents = ["player_"+str(i) for i in range(self.n_Veh)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_Veh))))
        self._agent_selector = agent_selector(self.agents)

        self.action_space = {i: spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float64) for i in self.agents}
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = {i: spaces.Box(low=-1000, high=1000, shape=(4,), dtype=np.float64) for i in self.agents}

        self.steps = 0
        self.max_step = 10000
        self.epoch = 0
        self.rewards = {i: 0 for i in self.agents}
        self.s_ = None

        self.add_new_vehicles_by_number(self.n_Veh, self.start_velocity)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def seed(self, seed=None):
        np.random.seed(seed)
    
    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    # 在一个圆内随机取若干个坐标点
    @staticmethod
    def get_distance(num, radius, center_x=0, center_y=0):
        distance = []
        for i in range(num):
            while True:
                x = np.random.uniform(-radius, radius)
                y = np.random.uniform(-radius, radius)
                if (x ** 2) + (y ** 2) <= (radius ** 2):
                    distance.append(np.hypot((int(x)+center_x), (int(y)+center_y)))
                    break

        distance[0] = 25
        distance[1] = 100

        distance.sort()
        return distance

    # 用于添加n个新的车辆对象
    def add_new_vehicles_by_number(self, n, start_velocity):
        start_distance = self.get_distance(n, self.radius)
        for i in range(n):
            self.vehicles.append(Vehicle(start_distance[i], start_velocity))

    def renew_positions(self):
        """
        This function update the position of each vehicle
        这个函数用于更新每一辆车的位置信息，每0.1s更新一次
        :return:
        """
        for i in range(len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_step * np.random.uniform(-1.0, 1.0)
            self.vehicles[i].distance = np.clip(self.vehicles[i].distance + delta_distance, 0, self.radius)

    def renew_channel(self):
        """
        This function updates all the channels including V2I channels
        更新V2I信道大尺度衰落的信道特性参数
        :return:
        """
        distances = [c.distance for c in self.vehicles]
        self.V2IChannels.update_distances(distances)
        self.V2IChannels.update_path_loss()

        # 获得新的大尺度衰落（w）
        self.V2I_channels_abs = self.V2IChannels.PathLoss   # 一维

    def renew_channels_fast_fading(self):
        """
        This function updates all the channels including V2I channels
        更新所有的信道包括V2I信道
        :return:
        """
        # 先更新所有信道的大尺度衰落参数
        self.renew_channel()
        # 更新小尺度衰落时的信道特性h（w）
        self.V2IChannels.update_fast_fading()

        """
        repeat(a, repeats, axis=None)，其中a为输入的数组，repeats为a中每个元素重复的次数
        axis代表重复数值的方向，axis=0代表y轴方向，axis=1代表x轴方向，axis=2代表z轴方向
        将V2I_channels_abs转化为二维，第二维为子频带数量。此时同一个V2I link的不同子频带的self.V2I_channels_abs值相同
        """
        v2i_channels_with_fast_fading = self.V2I_channels_abs[:, np.newaxis]
        # 计算出同时具有大尺度和小尺度衰落时的V2I信道特性（h（w）），即Gn，k（t）
        self.V2I_channels_with_fast_fading = self.V2IChannels.FastFading * np.sqrt(v2i_channels_with_fast_fading)

    def compute_reward(self, action):
        """
        Used for Training
        add the power dimension to the action selection
        :param action:
        :return:
        """
        # print(f"actions is:{action}")
        power_selection = np.array(action.copy()).reshape((self.n_Veh, 1))
        self.out_g = self.V2I_channels_with_fast_fading.reshape(self.n_Veh, 1)

        interference = np.zeros((self.n_Veh, 1))
        v2i_rate_list = np.zeros((self.n_Veh, 1))

        for n in range(self.n_Veh):
            if power_selection[n][0] != 0:
                for i in range(n+1, self.n_Veh):
                    if i != n:
                        interference[n][0] += (power_selection[i][0] *
                                                        self.V2I_channels_with_fast_fading[n] ** 2)
                interference[n][0] += self.sig2

        for n in range(self.n_Veh):
                if power_selection[n][0] != 0:
                    v2i_rate_list[n][0] = np.log2(1 + np.divide((power_selection[n][0] *
                                                                self.V2I_channels_with_fast_fading[n] ** 2),
                                                                interference[n][0]))

        self.cur_up = v2i_rate_list.copy()
        self.mutual_information += v2i_rate_list
        # print(f"self.V2I_channels_with_fast_fadin is:{self.V2I_channels_with_fast_fading}, v2i_rate_list is:{v2i_rate_list}, interference is:{interference}")
        cur_rate = self.accumulation_rate.copy()
        cur_rate[cur_rate>self.mutual_information] = 0



        

        # ee = np.divide(v2i_rate_list, (power_selection + self.PO))

        return cur_rate.reshape(-1)

    def step(self, action):
        """
        这个函数用于计算在训练时采用了动作action之后，环境的反馈reward
        :param action:
        :return:
        """
        action_temp = action.copy()
        action_temp = abs(action_temp)
        # print(action_temp)
        s = self.s_.copy()
        # rate = np.array([[0.5], [0.5]])
        rate = action_temp * 10
        # rate = action_temp[:,:1] * 50
        power = np.array([[1/self.Maximum_number_of_transmission_K], [1/self.Maximum_number_of_transmission_K]]) * self.PO
        # power = action_temp[:,1:] * self.PO
        self.accumulation_rate += rate
        tras_k = self.current_transmission_time.copy()

        cur_accumulation_power = self.accumulation_power + power
        cur_accumulation_power = np.clip(cur_accumulation_power, 0, self.PO)

        reward_sums = self.compute_reward(cur_accumulation_power - self.accumulation_power)  # 计算出采取了action之后各个信道的容量
        self.throughput += reward_sums.copy()

        outage_time = np.zeros((self.n_Veh, 1))
        for i in range(self.n_Veh):
            if self.current_transmission_time[i][0] < self.Maximum_number_of_transmission_K and reward_sums[i] <= 0:
                # print(f"self.Maximum_number_of_transmission_K is:{self.Maximum_number_of_transmission_K}")
                # print(f"{self.current_transmission_time[i][0] < self.Maximum_number_of_transmission_K}self.steps is:{self.steps}, self.current_transmission_time[{i}][0] is:{self.current_transmission_time[i][0]}")
                self.current_transmission_time[i][0] += 1
                self.accumulation_power[i][0] = cur_accumulation_power[i][0]
            else:
                if  reward_sums[i] <= 0 and self.current_transmission_time[i][0] >= self.Maximum_number_of_transmission_K:
                    outage_time[i][0] = 1
                    self.outage_num[i][0] += 1
                self.mutual_information[i][0] = 0
                self.accumulation_rate[i][0] = 0
                self.accumulation_power[i][0] = 0.
                # print(f"self.current_transmission_time[{i}][0] is:{self.current_transmission_time[i][0]}")
                self.current_transmission_time[i][0] = 1
                self.harq_time[i][0] += 1

        if self.training:
            for i in range(self.n_Veh):
                if self.harq_time[i][0] >= 1:
                    # reward_sums[i] = reward_sums[i] - self.lambd[i] * (outage_time[i][0] - self.phi[i]) - self.mu[i] * (self.epsilon - reward_sums[i]/self.avg_trans[i])
                    reward_sums[i] = sum(reward_sums) - self.lambd[i] * (outage_time[i][0] - self.phi[i]) - self.mu[i] * (self.epsilon - reward_sums[i]/self.avg_trans[i])
                    # reward_sums[i] = sum(reward_sums) - self.lambd[i] * (np.log10(1 + outage_time[i][0]) - np.log10(1 + self.phi[i])) - self.mu[i] * (self.epsilon - reward_sums[i]/self.avg_trans[i])
        
        done = False

        if self.steps >= self.max_step:
            done = True
            self.out_throughput = self.throughput
            self.avg_trans = self.max_step/self.harq_time
            self.epoch += 1
            if self.epoch % 2 == 0:
                for i in range(self.n_Veh):
                    if self.outage_num[i][0]/self.harq_time[i][0] - self.phi[i] > 0:
                        self.lambd[i] += self.beta[i] * (self.outage_num[i][0]/self.harq_time[i][0] - self.phi[i])
                        # self.lambd[i] += self.beta[i] * np.log10(self.outage_num[i][0]/self.harq_time[i][0] / self.phi[i])
                    else:
                        self.lambd[i] += 0
                    if self.epsilon > self.out_throughput[i]/self.max_step:
                        self.mu[i] += self.alpha[i] * (self.epsilon - self.out_throughput[i]/self.max_step)
                        # print(f"self.mu[{i}] is:{self.mu[i]}")
                    else:
                        self.mu[i] += 0
                print(f"lambd is:{self.lambd}, mu is:{self.mu}")

        # if self.harq_time.all() >= 1:
        #     self.s_ = np.hstack((self.accumulation_rate, self.mutual_information, self.out_g, self.outage_num/self.harq_time))
        # else:
        #     self.s_ = np.hstack((self.accumulation_rate, self.mutual_information, self.out_g, self.outage_num))
        out_gain = self.out_g.copy()
        out_gain = out_gain.reshape(-1)
        out_gain = np.array([out_gain, out_gain])
        self.s_ = np.hstack((self.accumulation_rate, self.mutual_information, out_gain))
        # 每采取一个action，更新一次环境
        self.renew_positions()
        self.renew_channels_fast_fading()

        info = {"harq_time" : self.harq_time, "outage_time": outage_time, "up": self.cur_up, "rews": reward_sums}
        # info = {"harq_time" : self.harq_time, "outage_time": outage_time}
        # if reward_sum == np.inf:
        #     print(action_temp, reward_sum)

        # reward_sums = [reward_sum for _ in range(len(self.agents))]

        
        self.steps += 1
        # print(f"accumulation_power is:{self.accumulation_power}, cur_accumulation_power is:{cur_accumulation_power}, s is:{s}, a is:{action_temp}, r is:{reward_sums}, s_ is:{self.s_}, tras_k is:{tras_k}")
        # return self.s_, np.array([sum(reward_sums), sum(reward_sums)]), done, info
        return self.s_, reward_sums, done, info

    def reset(self):
        # self.seed(self.se)
        np.random.seed(self.se)
        self.steps = 0

        # self.lambd = np.array([10**-3]*self.n_Veh)
        self.outage_num = np.zeros((self.n_Veh, 1))
        self.harq_time = np.zeros((self.n_Veh, 1))
        self.throughput = np.zeros(self.n_Veh)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        # self.vehicles = []
        # self.add_new_vehicles_by_number(self.n_Veh, self.start_velocity)
        self.renew_channels_fast_fading()
        self.mutual_information = np.zeros((self.n_Veh, 1))
        self.accumulation_rate = np.zeros((self.n_Veh, 1))
        self.current_transmission_time = np.ones((self.n_Veh, 1))
        self.s_ = np.hstack((self.accumulation_rate, self.mutual_information, self.V2I_channels_with_fast_fading[:, :1]))
        # return np.hstack((self.accumulation_rate, self.mutual_information, self.V2I_channels_with_fast_fading[:, :1], self.outage_num))
        out_gain = self.V2I_channels_with_fast_fading[:, :1].copy()
        out_gain = out_gain.reshape(-1)
        out_gain = np.array([out_gain, out_gain])
        return np.hstack((self.accumulation_rate, self.mutual_information, out_gain))