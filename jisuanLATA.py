import csv
import numpy as np

# 初始化一个列表来保存每行的和
sums = []

# 打开 CSV 文件并创建一个 CSV 读取器对象
with open('D:/CodeProject/PythonProject/NOMA_HARQ_DRL_FL/code/trainLTATFL.csv', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    
    # 逐行读取 CSV 文件中的数据
    for row in reader:
        array = np.fromstring(row[0][1:-1], sep=' ')
        # 计算数组元素的和
        total = np.sum(array)
        print(f"行数据: {array}, 和: {total}")

        sums.append(total)

# 将求和后的结果写入新的 CSV 文件
with open('sums_output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(sums)

print("求和后的数据已成功写入 sums_output.csv 文件中！")
