import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

import seaborn as sns


def load_and_process_loss(loss_tag, window_size=10):
    print(loss_tag)
    loss_values = np.array(loss_tag)

    # 计算每个窗口的最大值和最小值
    max_losses = [np.max(loss_values[i:i+window_size]) for i in range(0, len(loss_values), window_size)]
    min_losses = [np.min(loss_values[i:i+window_size]) for i in range(0, len(loss_values), window_size)]

    x_ticks = np.arange(0, len(loss_values), window_size)

    return x_ticks, max_losses, min_losses


reader = []
# 打开CSV文件
with open('train_data.csv', 'r') as f:
    readData = csv.reader(f)

    
    for row in readData:
        # 每一行数据都被存储为一个列表
        reader.append(row)

# time_steps = list(map(int, reader[0]))

time_steps, df1_train_reward, df1 = load_and_process_loss(list(map(float, reader[1])))
df1_train_reward = pd.DataFrame({'step': time_steps, 'reward': df1_train_reward, "name": ['K=3lr=1e-5'] * len(time_steps)})
df1 = pd.DataFrame({'step': time_steps, 'reward': df1, "name": ['K=3lr=1e-5'] * len(time_steps)})

time_steps, df2_train_reward, df2 = load_and_process_loss(list(map(float, reader[2])))
df2_train_reward = pd.DataFrame({'step': time_steps, 'reward': df2_train_reward, "name": ['K=3lr=5e-4'] * len(time_steps)})
df2 = pd.DataFrame({'step': time_steps, 'reward': df2, "name": ['K=3lr=5e-4'] * len(time_steps)})

time_steps, df3_train_reward, df3 = load_and_process_loss(list(map(float, reader[2])))
df3_train_reward = pd.DataFrame({'step': time_steps, 'reward': df3_train_reward, "name": ['K=3lr=1e-4'] * len(time_steps)})
df3 = pd.DataFrame({'step': time_steps, 'reward': df3, "name": ['K=3lr=1e-4'] * len(time_steps)})


df1 = df1.append(df1_train_reward)
df1.index = range(len(df1))


df2 = df2.append(df2_train_reward)
df2.index = range(len(df2))


df3 = df3.append(df3_train_reward)
df3.index = range(len(df3))

df = df1.append(df2)
df = df.append(df3)

df.index = range(len(df))

df = pd.DataFrame({'step': df['step'].values, 'reward': df['reward'].values / 100000, "name": df['name']})

data = [df1['step'].values, df1['reward'].values / 100000, df2['reward'].values / 100000, df3['reward'].values / 100000]


dataframe = pd.DataFrame(data)

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文为宋体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文为Times New Roman
plt.rcParams['axes.unicode_minus'] = False


p = sns.lineplot(data=df, x='step', y='reward', hue='name', style='name')
p.set_xlabel("Epoch", fontsize = 15)
p.set_ylabel("奖励", fontsize = 15)
plt.rcParams['text.usetex'] = True
plt.legend(labels=[r"$K=3, \upsilon = \alpha = 1 \times 10^{-5}$", r"$K=3, \upsilon = \alpha =5 \times 10^{-4}$", r"$K=3, \upsilon = \alpha = 1 \times 10^{-4}$"], fontsize = 15)
plt.show()