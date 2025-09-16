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

class V2IChannels:
    """Simulator of the V2I channels (V2I信道模拟器)"""

    def __init__(self, n_veh, n_rb):
        self.n_Veh = n_veh
        self.n_RB = n_rb
        self.distances = None
        self.PathLoss = None
        self.FastFading = None

    def update_distances(self, distances):
        self.distances = distances

    def update_path_loss(self):
        # 用于更新V2I之间的路径损失PL
        self.PathLoss = np.zeros(len(self.distances))
        for i in range(len(self.distances)):
            self.PathLoss[i] = 122.0 + 38.0 * np.log10(self.distances[i] / 1000)

    """
        初始化快（小规模）衰落的信道特征h(dB)，self.FastFading为一个二维数组，
        第二维是资源块的数量，即V2I link的每个子频带，都有着它自己的信道特性h(self.FastFading)
        二维数组的每一个元素都是实部和虚部都服从正太分布的一个复信特性
    """
    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_RB, self.n_Veh)) +
                            1j * np.random.normal(size=(self.n_RB, self.n_Veh)))
        self.FastFading = 20 * np.log10(np.abs(h))


class Vehicle:
    """
        Vehicle simulator: include all the information for a vehicle
        车辆类，每一辆车的信息包括：位置、方向、速度、它到的3个neighbors的距离、目的地
    """
    def __init__(self, start_distance, velocity):
        self.distance = start_distance  # 每个位置信号包含着（x, y）坐标的两个变量
        self.velocity = velocity  # 1m/s


class NOMA_Environ(gym.Env):
    """
        环境模拟器：（1）为agent提供状态s和反馈reward
        （2）根据agent所采取的动作action，环境会返回更新的状态s（t+1）
    """

    # 初始化环境参数
    def __init__(self, seed):
        np.random.seed(seed)
        self.se = seed
        self.time_step = 0.1  # 更新车辆位置信息的时间间隔
        self.vehicles = []  # 用于存储摆放在环境中的车辆对象（有Vehicle类生成）
        self.PO = 5
        self.sig2_dB = -174  # 环境高斯噪声功率（dBm）
        self.sig2 = 10**(self.sig2_dB/10)/1000  # 环境高斯噪声功率（W）
        self.delta_distance = []
        self.n_RB = 2  # 子频带的数量
        self.n_Veh = 2  # 车辆的数量
        self.state_dim = self.n_RB * self.n_Veh
        self.action_dim = self.n_RB * self.n_Veh
        self.action_bound = 1.0
        self.V2IChannels = V2IChannels(self.n_Veh, self.n_RB)  # 创建V2I信道对象
        self.reset_pointer = 0
        self.radius = 1000
        self.start_velocity = 1  # 用户的初始速度，此处为1m/s
        self.V2I_channels_abs = None
        self.V2I_channels_with_fast_fading = None

        self.agents = ["player_"+str(i) for i in range(self.n_Veh)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_Veh))))
        self._agent_selector = agent_selector(self.agents)

        self.action_space = {i: spaces.Box(low=-1, high=1, shape=(self.n_RB,), dtype=np.float64) for i in self.agents}
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = {i: spaces.Box(low=-1000, high=1000, shape=(self.n_RB, self.n_Veh,), dtype=np.float64) for i in self.agents}

        self.steps = 0
        self.rewards = {i: 0 for i in self.agents}
    
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

        # 获得新的大尺度衰落（dB）
        self.V2I_channels_abs = self.V2IChannels.PathLoss   # 一维

    def renew_channels_fast_fading(self):
        """
        This function updates all the channels including V2I channels
        更新所有的信道包括V2I信道
        :return:
        """
        # 先更新所有信道的大尺度衰落参数
        self.renew_channel()
        # 更新小尺度衰落时的信道特性h（dB）
        self.V2IChannels.update_fast_fading()

        """
        repeat(a, repeats, axis=None)，其中a为输入的数组，repeats为a中每个元素重复的次数
        axis代表重复数值的方向，axis=0代表y轴方向，axis=1代表x轴方向，axis=2代表z轴方向
        将V2I_channels_abs转化为二维，第二维为子频带数量。此时同一个V2I link的不同子频带的self.V2I_channels_abs值相同
        """
        v2i_channels_with_fast_fading = np.repeat(self.V2I_channels_abs[np.newaxis, :], self.n_RB, axis=0)
        # 计算出同时具有大尺度和小尺度衰落时的V2I信道特性（h（dB）），即Gk，n（t）
        self.V2I_channels_with_fast_fading = np.sqrt(v2i_channels_with_fast_fading) * self.V2IChannels.FastFading

    def compute_reward(self, action):
        """
        Used for Training
        add the power dimension to the action selection
        :param action:
        :return:
        """
        power_selection = np.array(action.copy()).reshape((self.n_RB, self.n_Veh))
        power_selection_db = np.zeros((self.n_RB, self.n_Veh))
        for k in range(self.n_RB):
            for n in range(self.n_Veh):
                if power_selection[k, n] != 0:
                    power_selection_db[k, n] = 10 * np.log10(power_selection[k, n])  # 功率选择

        interference = np.zeros((self.n_RB, self.n_Veh))
        v2i_rate_list = np.zeros((self.n_RB, self.n_Veh))

        for k in range(self.n_RB):
            for n in range(self.n_Veh):
                if power_selection[k, n] != 0:
                    for i in range(self.n_Veh):
                        if i != n:
                            interference[k, n] += 10 ** ((power_selection_db[k, i] *
                                                          self.V2I_channels_with_fast_fading[k, n] ** 2) / 10)
                    interference[k, n] += self.sig2

        for k in range(self.n_RB):
            for n in range(self.n_Veh):
                if power_selection[k, n] != 0:
                    v2i_rate_list[k, n] = np.log2(1 + np.divide(10 ** ((power_selection_db[k, n] *
                                                                self.V2I_channels_with_fast_fading[k, n] ** 2) / 10),
                                                                interference[k, n]))

        ee = np.divide(v2i_rate_list, (power_selection + self.PO))

        return np.sum(ee)

    def step(self, action):
        """
        这个函数用于计算在训练时采用了动作action之后，环境的反馈reward
        :param action:
        :return:
        """
        action_temp = action.copy()
        reward_sum = self.compute_reward(action_temp)  # 计算出采取了action之后各个信道的容量

        # 每采取一个action，更新一次环境
        self.renew_positions()
        self.renew_channels_fast_fading()

        s_ = self.V2I_channels_with_fast_fading.reshape((-1))

        return s_, reward_sum

    def reset(self):
        self.seed(self.se)
        self.steps = 0
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.vehicles = []
        self.add_new_vehicles_by_number(self.n_Veh, self.start_velocity)
        self.renew_channels_fast_fading()
        return self.V2I_channels_with_fast_fading.reshape((-1))


class XP_HARQ_NOMA_Environ(gym.Env):
    """Simulator of the V2I channels (V2I信道模拟器)"""

    def __init__(self, n_veh, n_rb):
        self.n_Veh = n_veh
        self.n_RB = n_rb
        self.distances = None
        self.PathLoss = None
        self.FastFading = None




class Environ(gym.Env):
    def __init__(self, power_db=20, max_K=3, epsilon=10**-3, beta=50, training=False, seed=123, name="none"):
        super(Environ, self).__init__()
        np.random.seed(seed)
        self.se = seed
        self.name = name
        self.Maximum_number_of_transmission_K = max_K
        self.sigma_squared = 1
        self.rho = 0.4
        self.lambd = 10**-3
        self.beta = beta
        self.training = training
        self.epsilon = epsilon
        self.outage_num = 0
        self.power_w = 10.**(power_db/10) * np.ones(self.Maximum_number_of_transmission_K+1)
        self.SNR_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
        self.rate_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
        self.current_transmission_time = 0
        self.h = 1/np.sqrt(2) * (np.random.normal() + 1j * np.random.normal())
        self.state_dim = 3
        # self.state_dim = 2
        self.action_dim = 1

        self.step_time = 1

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float64)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.state_dim,), dtype=np.float64)

        self.s = None
        self.r = 0
        self.a = None
        self.rate = None
        self.s_ = None
        self.mutual_information = 0
        self.cumulative_rate = 0.
        self.harq_time = 0
        self.traverse_capacity = 0

    def update_transmission_k_snr(self, k):
        # self.h = 1/np.sqrt(2) * (np.random.normal() + 1j * np.random.normal())
        self.h = self.rho * self.h + np.sqrt(1 - np.power(self.rho, 2)) * np.random.normal()
        transmission_k_snr = abs(self.h) ** 2 * self.power_w[k] / self.sigma_squared
        return transmission_k_snr

    def compute_reward(self, rate):
        self.mutual_information = 0
        self.cumulative_rate = sum(rate[:self.current_transmission_time])
        for k in range(1, self.current_transmission_time+1):
            self.mutual_information += np.log2(1 + self.SNR_transmission_K[-k])
        self.traverse_capacity = np.log2(1 + self.SNR_transmission_K[-1])
        if self.mutual_information < self.cumulative_rate:
            return np.array(0)
        else:
            return self.cumulative_rate

    def step(self, action):
        """
        这个函数用于计算在训练时采用了动作action之后，环境的反馈reward
        :param action:
        :return:
        """
        action_temp = action.copy()
        action_temp = action_temp * 10
        action_temp = abs(action_temp)
        # action_temp = np.clip(action_temp, 0, np.inf)
        # print(f"action_temp is:{action_temp}")

        # 给render使用
        self.s = self.s_

        self.SNR_transmission_K[:-1] = self.SNR_transmission_K[1:]
        self.SNR_transmission_K[-1] = self.update_transmission_k_snr(self.current_transmission_time)
        self.rate_transmission_K[self.current_transmission_time-1] = action_temp
        reward_sum = self.compute_reward(self.rate_transmission_K)  # 计算出采取了action之后各个信道的容�?

        # 给render使用
        self.r = reward_sum
        self.a = action_temp
        self.rate = self.rate_transmission_K.copy()


        # 每采取一个action，更新一次环境
        outage_time = 0
        if self.current_transmission_time < self.Maximum_number_of_transmission_K and reward_sum <= 0:
            self.current_transmission_time += 1
        else:
            if reward_sum <= 0 and self.current_transmission_time >= self.Maximum_number_of_transmission_K:
                outage_time = 1
                self.outage_num += 1
            self.rate_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
            self.mutual_information = 0.
            self.cumulative_rate = 0.
            self.current_transmission_time = 1
            self.harq_time += 1
        
        if self.training:
            if self.harq_time >= 1:
                reward_sum = reward_sum - self.lambd * outage_time

        if self.step_time % 10 == 0:
            if self.outage_num/self.harq_time - self.epsilon > 0:
                self.lambd += self.beta * np.log10(self.outage_num/self.harq_time / self.epsilon)
            else:
                self.lambd += 0
            # print(f"lambd is:{self.lambd}, p_out_t is:{self.outage_num/self.harq_time}, gap is:{self.outage_num/self.harq_time - self.epsilon}, all_item is:{self.beta * np.log10(self.outage_num/self.harq_time / self.epsilon)}")
            # self.lambd += max(self.beta * np.log10(self.outage_num/self.harq_time - self.epsilon), 0)
            # print(f"lambd:{self.lambd}, out_time:{outage_time}, after item:{(self.epsilon - (self.outage_num/self.harq_time))}, all item:{self.lambd*(self.epsilon - (self.outage_num/self.harq_time))}")
        
            
        if self.step_time >= 100000:
            # if self.name == "test":
            #     print(f"self.step_time is:{self.step_time}")
            # else:
            #     print(self.step_time)
            done = True
        else:
            done = False
            
        self.step_time += 1

        info = {"harq_time" : self.harq_time, "outage_time": outage_time, "traverse_capacity": self.traverse_capacity}

        self.s_ = np.array([self.cumulative_rate, self.mutual_information, self.SNR_transmission_K[-1]])
        # self.render()
        return self.s_, reward_sum, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        np.random.seed(self.se)
        self.step_time = 1
        self.harq_time = 0
        self.lambd = 10**-3
        self.outage_num = 0
        self.h = 1/np.sqrt(2) * (np.random.normal() + 1j * np.random.normal())
        self.SNR_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
        for i in range(self.Maximum_number_of_transmission_K):
            self.SNR_transmission_K[i] = self.update_transmission_k_snr(i)
        self.rate_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
        self.current_transmission_time = 1
        return np.array([0., 0., self.SNR_transmission_K[-1]])

    def render(self, mode="human"):
        print(f"s is:{self.s}, a is:{self.a}, r is:{self.r}, rate is:{self.rate}, s_ is:{self.s_}")


class IR_Environ(Environ):
    def __init__(self, power_db=20, max_K=3, epsilon=10**-3, training=False):
        super(IR_Environ, self).__init__()
        self.Maximum_number_of_transmission_K = max_K
        self.power_w = 10.**(power_db/10) * np.ones(self.Maximum_number_of_transmission_K+1)
        self.SNR_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
        self.rate_transmission_K = 0
        self.epsilon = epsilon
        self.training = training
        self.state_dim = len(self.SNR_transmission_K)

        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float64)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.state_dim,), dtype=np.float64)

    def compute_reward(self, rate, k):
        self.SNR_transmission_K[:-1] = self.SNR_transmission_K[1:]  # 将CSI左移一�?
        self.SNR_transmission_K[-1] = self.update_transmission_k_snr(k)  # 更新最新的CSI
        mutual_information = 0
        for i in range(1, k+1):
            mutual_information += np.log2(1 + self.SNR_transmission_K[-i])
        if mutual_information < rate[0]:
            return np.array(0)
        else:
            return rate[0]

    def step(self, action):
        """
        这个函数用于计算在训练时采用了动作action之后，环境的反馈reward
        :param action:
        :return:
        """
        action_temp = action.copy()
        action_temp = action_temp * 10
        action_temp = abs(action_temp)
        # action_temp = np.clip(action_temp, 0, np.inf)
        # print(f"action_temp is:{action_temp}")

        # 给render使用
        self.s = self.SNR_transmission_K.copy()

        outage_time = 0
        for i in range(1, self.Maximum_number_of_transmission_K+1):
            reward_sum = self.compute_reward(action_temp, i)  # 计算出采取了action之后各个信道的容
            if reward_sum > 0:
                break
            else:
                if i >= self.Maximum_number_of_transmission_K:
                    outage_time = 1
                    self.outage_num += 1

        # 给render使用
        self.r = reward_sum
        self.a = action_temp
        self.rate = self.rate_transmission_K.copy()
        self.rate[0] = action_temp

        self.step_time += i
        self.harq_time += 1

        if self.training:
            reward_sum = reward_sum - self.lambd * outage_time

        if self.step_time % 10 == 0:
            if self.outage_num/self.harq_time - self.epsilon > 0:
                self.lambd += -1 * self.beta * np.log10(self.outage_num/self.harq_time - self.epsilon)
            else:
                self.lambd += 0
            # self.lambd += max(self.beta * np.log10(self.outage_num/self.harq_time - self.epsilon), 0)
            # print(f"lambd:{self.lambd}, out_time:{outage_time}, after item:{(self.epsilon - (self.outage_num/self.harq_time))}, all item:{self.lambd*(self.epsilon - (self.outage_num/self.harq_time))}")

        if self.step_time >= 6000:
            done = True
        else:
            done = False
        info = {"outage_time": outage_time, "harq_time": self.harq_time, "duration": i}

        # self.render()
        return self.SNR_transmission_K, reward_sum, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        self.step_time = 1
        self.harq_time = 0
        self.lambd = 1
        self.outage_num = 0
        self.h = 1/np.sqrt(2) * (np.random.normal() + 1j * np.random.normal())
        self.SNR_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
        for i in range(self.Maximum_number_of_transmission_K):
            self.SNR_transmission_K[i] = self.update_transmission_k_snr(i)
        self.rate_transmission_K = np.zeros(self.Maximum_number_of_transmission_K)
        self.current_transmission_time = 1
        return self.SNR_transmission_K

    def render(self, mode="human"):
        print(f"s is:{self.s}, a is:{self.a}, r is:{self.r}, rate is:{self.rate}, s_ is:{self.SNR_transmission_K}")
