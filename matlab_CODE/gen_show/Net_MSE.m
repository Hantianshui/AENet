clc
clear all;
close all;
% 导入测试数据
% load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_oriSig_Array_10_20_target_1_5dB.mat")
% % load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeSNR\Test_oriSig_Array_10_20_theta_-60_52_target_2_30dB.mat")

load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_oriSig_Array_10_20_theta_-60_52_target_2_30dB.mat")
% 获取复数矩阵
% LoriSig = complex(L_test_R,L_test_I);
% HoriSig = complex(H_test_R,H_test_I);
NoriSig = complex(predict_H_R,predict_H_I);

oriSig = NoriSig;

% 获取阵列的阵元数
N = size(oriSig,2);
target_num = 2;
% 记录每个样本的角度，两个目标
% tra_theta = zeros(length(one_hot_encoding),target_num);
smaple_num = size(test_angles, 1);
real_theta = test_angles;
% 传统方法扫描范围
scan_theta = -60:1:60;

% 定义导向矢量矩阵
steering_matrix = zeros(N, length(scan_theta));
for idx = 1:length(scan_theta)
    theta = scan_theta(idx);
    steering_matrix(:, idx) = exp(1i * 2 * pi * (0:N-1)' * 0.5 * sind((theta)));
end

for i = 1:smaple_num
    disp(i);
    R = oriSig(i,:);
    B_CBF = abs(R*steering_matrix);
%     figure(1)
%     plot(B_CBF)
    [~, peak_indices] = findpeaks(B_CBF, 'SortStr', 'descend', 'NPeaks', 2);
    theta_peaks = scan_theta(peak_indices);
    CBF_theta(i,:) = sort(round(theta_peaks));
end

tra_theta = CBF_theta;


% 角度真值，从小到大排序
actual_theta = sort(real_theta,2);

% 计算每个样本的绝对误差
errors = abs(actual_theta - tra_theta);

% 计算均方误差 (MSE)
squared_errors = errors .^ 2;
mse = mean(squared_errors(:)); % 将所有样本的平方误差展平并计算平均值

% 计算均方根误差 (RMSE)
rmse = sqrt(mse);

% 打印 RMSE
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmse);


% 统计不同阈值的正确率
% 误差阈值列表
thresholds = [0,2,4];

% 初始化结果存储
accuracies = zeros(length(thresholds), 1);
correct_predictions_counts = zeros(length(thresholds), 1);

% 遍历每个阈值计算准确率
for i = 1:length(thresholds)
    threshold = thresholds(i);

    % 统计误差在阈值内的样本数量
    correct_predictions = sum(all(errors <= threshold,2));
    correct_predictions_counts(i) = correct_predictions;

    % 计算准确率
    accuracy = correct_predictions / size(actual_theta, 1);
    accuracies(i) = accuracy;

    % 打印结果
    fprintf('阈值: %.1f degrees\n', threshold);
    fprintf('正确判断的样本数: %d\n', correct_predictions);
    fprintf('正确率: %.2f%%\n', accuracy * 100);

      % 找到两个角度都满足阈值条件的样本
    valid_samples = all(errors <= threshold, 2);
    
    % 筛选出满足条件的误差
    valid_errors = errors(valid_samples, :);
    
    % 计算均方误差 (MSE)
    squared_errors = valid_errors .^ 2;
    mse = mean(squared_errors(:)); % 将所有满足条件的样本的平方误差展平并计算平均值
    
    % 打印 MSE
    fprintf('Mean Squared Error (MSE) for threshold %.2f: %.4f\n', threshold, mse);
        % 计算均方根误差 (RMSE)
    rmse = sqrt(mse);
    % 打印 MSE
    fprintf('Mean Squared Error (RMSE) for threshold %.2f: %.4f\n', threshold, rmse);

end

%
% figure(1)
% plot(1:smaple_num, tra_theta(:,1),'ro');
% hold on;
% plot(1:smaple_num, tra_theta(:,2),'bo');
% legend('target1','target2')
% xlabel('样本数')
% ylabel('角度')
% title('CBF结果')
% 
% figure(2)
% plot(1:smaple_num, actual_theta(:,1),'ro');
% hold on;
% plot(1:smaple_num, actual_theta(:,2),'bo');
% legend('target1','target2')
% xlabel('样本数')
% ylabel('角度')
% title('ground truth')
% 
% figure(3)
% plot(1:smaple_num, errors(:,1),'ro');
% hold on;
% plot(1:smaple_num, errors(:,2),'bo');
% legend('target1','target2')
% xlabel('样本数')
% ylabel('角度')
% title('误差')



