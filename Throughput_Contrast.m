M = csvread('D:\CodeProject\PythonProject\NOMA_HARQ_DRL_FL\code\throughput_data.csv')
y0 = M(2, :)
y1 = M(3, :)
y2 = M(4, :)
y3 = M(5, :)
x = [10,15,20,25,30,35];
plot(x, y0, 'b-.', x, y1, '->', x, y2, '-*', x, y3, '--s', 'linewidth', 2)
legend('Coop-MADRL','FRL', 'DRL', 'Comp-MADRL', 'Times New Roman')
xlabel('\fontsize{12}SNR \fontsize{15} [dB]', 'Interpreter', 'tex', 'FontName', 'Times New Roman');
ylabel('LTAT$~\eta~$[bps/Hz]','interpreter','latex')

y0 = M(2, :);
y1 = M(3, :);
y2 = M(4, :);
y3 = M(5, :);
x = [10, 15, 20, 25, 30, 35];

bar(x, [y0' y1' y2' y3'], 'grouped');
legend('Coop-MADRL', 'FRL', 'DRL', 'Comp-MADRL');
xlabel('\fontsize{12}SNR \fontsize{15} [dB]', 'Interpreter', 'tex', 'FontName', 'Times New Roman');
ylabel('LTAT$~\eta~$[bps/Hz]','interpreter','latex');