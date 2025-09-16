% 导入数据
data = readtable('train_data.csv');

% 定义平滑函数
function smoothed = smooth(data, weight)
    scalar = data.reward;
    last = scalar(1);
    smoothed = zeros(size(scalar));
    for i = 1:length(scalar)
        smoothed_val = last * weight + (1 - weight) * scalar(i);
        smoothed(i) = smoothed_val;
        last = smoothed_val;
    end
end

% 平滑数据
df1_train_reward = data(:,1);
df1_train_reward = [df1_train_reward, smooth(data(:,2), 0.98)];

% 画图
figure
plot(df1_train_reward(:,1), df1_train_reward(:,2))
xlabel('Epoch')
ylabel('Reward')
legend('K=3 lr=1e-3', 'XP-HARQ K=3 lr=5e-4', 'XP-HARQ K=3 lr=1e-4')
