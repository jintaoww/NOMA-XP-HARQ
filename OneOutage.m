M = csvread('D:\CodeProject\PythonProject\NOMA_HARQ_DRL_FL\code\Oneoutage_data.csv')
y1 = M(2, :)
y2 = M(3, :)
y3 = M(4, :)
y4 = M(5, :)
y5 = M(6, :)
y9 = M(7, :)
x = [10,15,20,25,30, 35];
semilogy(x, y1, 'b-.', x, y2, '->', x, y3, '-*', x, y4, '--s',x, y5, 'k-h', x, y9,'linewidth', 2)
axis([10,35,3e-4,3e-1])
legend('Coop-MADRL','FRL', 'DRL', 'Comp-MADRL', 'NOMA', 'Outeage probability constaints','interpreter','latex')
%xlabel('SNR [dB]', 'fontsize', [10, 20]);
%text(0.5, 0.95, '[dB]', 'units', 'normalized', 'fontsize', 15);
%xlabel('SNR');
%text(1.05, 0.5, '[dB]', 'units', 'normalized', 'rotation', 90, 'fontsize', 15);
%set(gca, 'xlabel', text('string', 'SNR [dB]', 'fontsize', 10));
xlabel('\fontsize{12}SNR \fontsize{15} [dB]', 'Interpreter', 'tex', 'FontName', 'Times New Roman');

%xlabel('\textnormal{SNR} [\textnormal{dB}]','Interpreter','latex', 'FontSize', 14)
%hxlabel = xlabel('SNR');
%htext = text(hxlabel.Position(1), hxlabel.Position(2), '[dB]');
%set(htext, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'fontsize', 15);
%set(hxlabel, 'fontsize', 14);
ylabel('Outage probability $~p^{out}_{1,K}~$','Interpreter','latex')
%在matlab的同一行中用14的大小显示SNR，用15的大小显示[dB]
%text('interpreter' , 'latex' , 'string' , '\fontsize{12}{0}\selectfont$xxx$\fontsize{8}{0}\selectfont$yyyy$' );