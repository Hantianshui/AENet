clc
clear all;
close all;
% 导入测试数据
% load("C:\Users\buaa\Desktop\毕设相关\DOA估计\matlab\TestData_covSig_2Target_changeSNR\Test_covSig_Array_10_theta_-60_60_snap_100_-30dB.mat")
load("C:\Users\buaa\Desktop\毕设相关\DOA估计\matlab\TestData_covSig_2Target_changeSNR\Test_offgrid_covSig_Array_10_theta_-59.3_58.7_snap_100_30dB.mat")

% 获取复数矩阵
covSig = complex(covSig_R_all,covSig_I_all);
oriSig = complex(oriSig_R_all(:,:,1),oriSig_I_all(:,:,1));
% 获取阵列的阵元数
N = size(covSig,2);
target_num = 2;
% 记录每个样本的角度，两个目标
% tra_theta = zeros(length(one_hot_encoding),target_num);
smaple_num = size(real_theta, 1);
% 扫描范围
scan_theta = -60:1:60;
theta_grid = 0.1;
array_num = N;

CAPON_theta = zeros(smaple_num,target_num);
% Capon algorithm
angles = -90:1:90;
P_capon = zeros(size(scan_theta));
for i = 1:smaple_num
    disp(i);
    R = squeeze(covSig(i,:,:));

    for j = 1:length(scan_theta)
        steering_vector = exp(-1j*2*pi*0.5*(0:N-1)'*sind(scan_theta(j)));
        P_capon(j) = 1 / (steering_vector' / R * steering_vector);
    end

    % Convert to dB scale
    P_capon = 10*log10(abs(P_capon) / max(abs(P_capon)));
    % P_capon = abs(P_capon);

    %     [Pmusic, angle] = music_grid(R, array_num, target_num, theta_grid);
    %     THETA = Rmusic(R, array_num, target_num);
    %     [B_CBF,THETA] = CBF(R,array_num,theta_grid);
    [~, peak_indices] = findpeaks(P_capon, 'SortStr', 'descend', 'NPeaks', 2);
    theta_peaks = scan_theta(peak_indices);
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