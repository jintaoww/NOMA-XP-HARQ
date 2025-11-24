# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2022年04月15日
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(128, 64)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(64, 32)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(32, n_actions)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        output = self.fc4(x)
        return output


class ActorNet(nn.Module):
    def __init__(self, state_dim, a_dim, action_bound=1):
        super(ActorNet, self).__init__()

        self.action_bound = action_bound

        self.fc1 = nn.Linear(np.prod(state_dim), 128)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(128, 64)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(64, 32)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(32, np.prod(a_dim))
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        # print(f"obs is:{obs}")
        batch = obs.shape[0]
        x = f.relu((self.fc1(obs.view(batch, -1))))
        x = f.relu((self.fc2(x)))
        x = f.relu((self.fc3(x)))
        x = f.sigmoid(self.fc4(x))
        # x = f.relu(self.fc4(x))
        # x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        # x = torch.clamp(x, min=1e-3, max=1)
        return x, state


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(np.prod(s_dim)+np.prod(a_dim), 128)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(128, 64)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(64, 32)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(32, 1)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, s, a):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float)
        batch = s.shape[0]
        x = torch.cat([s.view(batch, -1), a.view(batch, -1)], dim=1)
        x = f.relu((self.fc1(x)))
        x = f.relu((self.fc2(x)))
        x = f.relu((self.fc3(x)))
        x = self.fc4(x)
        return x
