clc;
clear all;
close all;

% 加载数据
load('CNNinterval_accuracy.mat')
CNNinterval_accuracy = interval_accuracy;
load('Hinterval_accuracy.mat')
Hinterval_accuracy = interval_accuracy;
load('Linterval_accuracy.mat')
Linterval_accuracy = interval_accuracy;

% 定义角度间隔
angle_intervals = 1:12;

% 绘制图形
figure;
hold on;
plot(angle_intervals, CNNinterval_accuracy * 100, '-o', 'LineWidth', 1.5);
plot(angle_intervals, Hinterval_accuracy * 100, '-s', 'LineWidth', 1.5);
plot(angle_intervals, Linterval_accuracy * 100, '-d', 'LineWidth', 1.5);

% 图例和标签
legend('AENetCBF Accuracy', 'HArrayCBF Accuracy', 'LArrayCBF Accuracy');
xlabel('Angle Interval (degrees)');
ylabel('Accuracy (%)');
title('Accuracy for Different Angle Intervals');
xticks(angle_intervals); % 设置x轴刻度为1到12
grid on;
hold off;

% 保存图像
saveas(gcf, 'interval_accuracy_comparison.png');
