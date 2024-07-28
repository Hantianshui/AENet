clc;
clear all;
close all;

% 导入测试数据
% load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeSNR\Test_oriSig_Array_10_20_theta_-60_60_target_2_20dB.mat")

% % 获取复数矩阵
% LoriSig = complex(L_test_R, L_test_I);
% HoriSig = complex(H_test_R, H_test_I);
% 
% oriSig = LoriSig;

load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_oriSig_Array_10_20_theta_-60_60_target_2_20dB.mat")

NoriSig = complex(predict_H_R,predict_H_I);

oriSig = NoriSig;

% 获取阵列的阵元数
N = size(oriSig, 2);
target_num = 2;
% test_angles = real_theta;
% 记录每个样本的角度，两个目标
smaple_num = size(test_angles, 1);
real_theta = test_angles;

% 传统方法扫描范围
scan_theta = -60:1:60;

% 定义导向矢量矩阵
steering_matrix = zeros(N, length(scan_theta));
for idx = 1:length(scan_theta)
    theta = scan_theta(idx);
    steering_matrix(:, idx) = exp(1i * 2 * pi * (0:N-1)' * 0.5 * sind(theta));
end

CBF_theta = zeros(smaple_num, target_num);

for i = 1:smaple_num
    R = oriSig(i, :);
    B_CBF = abs(R * steering_matrix);
    [~, peak_indices] = findpeaks(B_CBF, 'SortStr', 'descend', 'NPeaks', 2);
    theta_peaks = scan_theta(peak_indices);
    CBF_theta(i, :) = sort(round(theta_peaks));
end

tra_theta = CBF_theta;

% 角度真值，从小到大排序
actual_theta = sort(real_theta, 2);

% 计算每个样本的绝对误差
errors = abs(actual_theta - tra_theta);

% 计算均方误差 (MSE)
squared_errors = errors .^ 2;
mse = mean(squared_errors(:)); % 将所有样本的平方误差展平并计算平均值

% 计算均方根误差 (RMSE)
rmse = sqrt(mse);

% 打印 RMSE
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmse);

% 初始化不同间隔的统计
unique_intervals = unique(labels);
interval_accuracy = zeros(length(unique_intervals), 1);

% 统计每种间隔的正确估计比例
for j = 1:length(unique_intervals)
    interval = unique_intervals(j);
    interval_samples = labels == interval;
    threshold = interval / 2; % 动态设置阈值为角度间隔的一半

    % 统计误差在阈值内的样本数量
    errors_interval = errors(interval_samples, :);
    correct_predictions = sum(all(errors_interval <= threshold, 2));
    
    % 计算准确率
    interval_accuracy(j) = correct_predictions / sum(interval_samples);

    fprintf('角度间隔: %d degrees\n', interval);
    fprintf('阈值: %.1f degrees\n', threshold);
    fprintf('正确估计的样本数: %d\n', correct_predictions);
    fprintf('样本总数: %d\n', sum(interval_samples));
    fprintf('正确率: %.2f%%\n\n', interval_accuracy(j) * 100);
end

% 保存每个间隔的正确估计比例
save('interval_accuracy.mat', 'interval_accuracy', 'unique_intervals');

% 显示间隔正确估计比例
figure;
bar(unique_intervals, interval_accuracy * 100);
xlabel('Angle Interval (degrees)');
ylabel('Accuracy (%)');
title('Accuracy for Different Angle Intervals');
grid on;
