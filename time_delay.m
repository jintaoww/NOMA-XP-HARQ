M = csvread('D:\CodeProject\PythonProject\NOMA_HARQ_DRL_FL\code\throughput_data.csv')
%y0 = 10^6./(10^7.*M(2, :))
y1 = 10^6./(10^7.*M(2, :))
y2 = 10^6./(10^7.*M(3, :))
y3 = 10^6./(10^7.*M(4, :))
y4 = 10^6./(10^7.*M(5, :))
x = [10,15,20,25,30,35];
plot(x, y1, 'b-.', x, y2, '->', x, y3, '-*', x, y4, '--s', 'linewidth', 2)
legend('Cooperative multi-agent','Federated multi-agent', 'Centralized', 'Competitive multi-agent', 'Times New Roman')
xlabel('\fontsize{17}SNR \fontsize{20} [dB]', 'Interpreter', 'tex', 'FontName', 'Times New Roman');
ylabel('transmission delay$~\mathcal T~$[s]','interpreter','latex')