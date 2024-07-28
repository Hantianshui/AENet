% function [output1, output2, ...] = functionName(input1, input2, ...)
function [B_CBF,THETA] = CBF(R,N,theta_grid)
%% 接收阵列参数
d = 0.5;   %为了避免栅瓣，这是对阵元间距最严格的要求
M = N;     %阵元数、均匀线列阵
% 产生阵列索引值，用来进行相位补偿向量的产生
% dd = (0 : N - 1) * d;
% disp(["阵元数：" num2str(M) ",分辨率为：" num2str(2/80/pi*180)]);
% disp(['阵元数：', num2str(M), ', 理论分辨率为：', num2str(2/M/pi*180)]);
% 
% disp(['阵元数：', num2str(M), ', 理论波束宽度为：', num2str(2*asind(2/M))]);
% 
% disp(['阵元数：', num2str(M), ', 理论阵列主波束宽带：', num2str(0.886/M*2*180/pi)]);

%% CBF
% 设置扫描角度区间以及步进角度值
THETA = -90:theta_grid:90;
X = R;
b_CBF = zeros(1,length(THETA));
for i = 1:length(THETA)
    a = exp(-1j*2*pi*d*sin(THETA(i)/180*pi)*(0:M-1)');%这是相位补偿向量
%     a = kron(a,a);
    Y = a'*X;        %对接收到的信号做相位补偿并求和，Y是1*N的向量
    b_CBF(i) = Y*Y'/N ;    %求平均能量
end
%[b I]=max(abs(b_CBF));
% 以dB形式展示
B_CBF = 10*log10(abs(b_CBF)/max(abs(b_CBF)));

% plot(THETA,B_CBF,'LineWidth',1.5);
end