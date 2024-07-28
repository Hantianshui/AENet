clc
clear all;
close all;
% 导入测试数据
% load("C:\Users\buaa\Desktop\毕设相关\DOA估计\matlab\TestData_covSig_2Target_changeSNR\Test_covSig_Array_10_theta_-60_60_snap_100_-30dB.mat")
% load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_single_changeSNR\Test_oriSig_Array_10_20_target_1_20dB.mat")

load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeSNR\Test_oriSig_Array_10_20_theta_-60_52_target_2_0dB.mat")
% 获取复数矩阵
LoriSig = complex(L_test_R,L_test_I);
HoriSig = complex(H_test_R,H_test_I);
% 获取阵列的阵元数
N = size(HoriSig,2);
target_num = 2;
% 记录每个样本的角度，两个目标
% tra_theta = zeros(length(one_hot_encoding),target_num);
smaple_num = size(test_angles, 1);
% 扫描范围
scan_theta = -90:1:90;
theta_grid = 1;
array_num = N;

CAPON_theta = zeros(smaple_num,target_num);
% Capon algorithm
angles = -90:1:90;
P_capon = zeros(size(angles));
for i = 1:smaple_num
    disp(i);
    R = squeeze(covSig(i,:,:));


    [B_CBF,THETA] = CBF(R,array_num,theta_grid);
    [~, peak_indices] = findpeaks(B_CBF, 'SortStr', 'descend', 'NPeaks', 2);
    theta_peaks = THETA(peak_indices);
    CAPON_theta(i,:) = round(theta_peaks);
end


% 角度真值，从小到大排序
actual_theta = sort(real_theta,2);

% 计算每个样本的绝对误差
errors = abs(actual_theta - CAPON_theta);

% 计算均方误差 (MSE)
squared_errors = errors .^ 2;
mse = mean(squared_errors(:)); % 将所有样本的平方误差展平并计算平均值

% 计算均方根误差 (RMSE)
rmse = sqrt(mse);

% 打印 RMSE
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmse);


% 统计不同阈值的正确率
% 误差阈值列表
thresholds = [1, 3, 5];

% 初始化结果存储
accuracies = zeros(length(thresholds), 1);
correct_predictions_counts = zeros(length(thresholds), 1);

% 遍历每个阈值计算准确率
for i = 1:length(thresholds)
    threshold = thresholds(i);

    % 统计误差在阈值内的样本数量
    correct_predictions = sum(all(errors <= threshold, 2));
    correct_predictions_counts(i) = correct_predictions;

    % 计算准确率
    accuracy = correct_predictions / size(actual_theta, 1);
    accuracies(i) = accuracy;

    % 打印结果
    fprintf('阈值: %.1f degrees\n', threshold);
    fprintf('正确判断的样本数: %d\n', correct_predictions);
    fprintf('正确率: %.2f%%\n', accuracy * 100);
end