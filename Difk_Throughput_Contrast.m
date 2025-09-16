M = csvread('D:\CodeProject\PythonProject\NOMA_HARQ_DRL_FL\code\difk_difAl_throughput_data.csv')
% 创建柱状图
x_values = 2:5;
bar(x_values, M);

% 添加标题和标签（可选）
xlabel('Maximum number of transmissions $K$','interpreter','latex');                                                                                                                                                                            
ylabel('LTAT$~\eta~$[bps/Hz]','interpreter','latex');

legend('Comp-MADRL', 'Coop-MADRL', 'DRL', 'FRL');
