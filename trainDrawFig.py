import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from matplotlib import font_manager

import seaborn as sns

def smooth(data, x='step', y='reward', weight=0.98):

    scalar = data[y].values
    print(scalar)
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return pd.DataFrame({x: data[x].values, y: smoothed, "name": data['name']})



reader = []
# 打开CSV文件
with open('train_data.csv', 'r') as f:
    readData = csv.reader(f)

    
    for row in readData:
        # 每一行数据都被存储为一个列表
        reader.append(row)

time_steps = list(map(int, reader[0]))

df1_train_reward = [(i, float(reader[1][i]), 'K=3lr=1e-5') for i in range(len(time_steps))]
df1_train_reward = pd.DataFrame(df1_train_reward, columns=['step', 'reward', 'name'])

df2_train_reward = [(i, float(reader[2][i]), 'K=3lr=5e-4') for i in range(len(time_steps))]
df2_train_reward = pd.DataFrame(df2_train_reward, columns=['step', 'reward', 'name'])

df3_train_reward = [(i, float(reader[3][i]), 'K=3lr=1e-4') for i in range(len(time_steps))]
df3_train_reward = pd.DataFrame(df3_train_reward, columns=['step', 'reward', 'name'])

weight2 = 0
df1 = smooth(df1_train_reward)
df1_train_reward = smooth(df1_train_reward, weight=weight2)
df1 = df1.append(df1_train_reward)
df1.index = range(len(df1))

df2 = smooth(df2_train_reward)
df2_train_reward = smooth(df2_train_reward, weight=weight2)
df2 = df2.append(df2_train_reward)
df2.index = range(len(df2))

df3 = smooth(df3_train_reward)
df3_train_reward = smooth(df3_train_reward, weight=weight2)
df3 = df3.append(df3_train_reward)
df3.index = range(len(df3))

df = df1.append(df2)
df = df.append(df3)

df.index = range(len(df))

df = pd.DataFrame({'step': df['step'].values, 'reward': df['reward'].values, "name": df['name']})

data = [df1['step'].values, df1['reward'].values, df2['reward'].values, df3['reward'].values]


dataframe = pd.DataFrame(data)

# 设置全局字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文为宋体
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文为Times New Roman
plt.rcParams['axes.unicode_minus'] = True


p = sns.lineplot(data=df, x='step', y='reward', hue='name', style='name')

p.set_xlabel("Epoch", fontsize = 20)
p.set_ylabel("LATA", fontsize = 20)
plt.rcParams['text.usetex'] = True
# plt.legend(labels=[r"Comp-MADRL", r"Coop-MADRL", r"FRL"], fontsize = 15)
plt.legend(labels=[r"$K=3, \upsilon = \alpha = 1 \times 10^{-5}$", r"$K=3, \upsilon = \alpha =5 \times 10^{-4}$", r"$K=3, \upsilon = \alpha = 1 \times 10^{-4}$"], fontsize = 15)
plt.show()


    



# # 为了模拟误差，我们创建一些随机的上下界
# beta_happo_upper = beta_happo + np.random.normal(0.1, 0.02, 100)
# beta_happo_lower = beta_happo - np.random.normal(0.1, 0.02, 100)

# plt.figure(figsize=(10, 6))

# # 绘制线条
# plt.plot(time_steps, beta_happo, color='red', label='Beta-HAPPO (Proposed)')
# plt.fill_between(time_steps, beta_happo_lower, beta_happo_upper, color='red', alpha=0.1)  # 添加阴影

# # 其他线条和阴影可以类似地添加

# # 添加图例
# plt.legend()

# # 添加标题和标签
# plt.title('Average energy consumption of MVUs')
# plt.xlabel('Time steps')
# plt.ylabel('Average energy consumption (ml)')

# # 显示图形
# plt.show()
