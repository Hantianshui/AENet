clc
clear all;
close all;
% 导入测试数据
% load("C:\Users\buaa\Desktop\毕设相关\DOA估计\matlab\TestData_covSig_2Target_changeSNR\Test_offgrid_covSig_Array_10_theta_-59.3_58.7_snap_100_20dB.mat")
load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeSNR\Test_oriSig_Array_10_20_theta_-60_52_target_2_snap_100_0dB.mat")
% 获取复数矩阵

% load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_oriSig_Array_10_20_theta_-60_52_target_2_snap_100_30dB.mat")
% 获取复数矩阵
LoriSig = complex(L_test_R,L_test_I);
HoriSig = complex(H_test_R,H_test_I);
% NoriSig = complex(predict_H_R,predict_H_I);
coriSig = LoriSig;
% 获取阵列的阵元数
N = size(coriSig,2);
target_num = 2;
real_theta = test_angles;
% 记录每个样本的角度，两个目标
% tra_theta = zeros(length(one_hot_encoding),target_num);
smaple_num = size(real_theta, 1);
total_num_samples = size(coriSig, 1);
% 扫描范围
scan_theta = -60:1:60;
grid_step = 0.5;
array_num = N;

MUSIC_theta = zeros(smaple_num,target_num);

for i = 1:smaple_num
    X = squeeze(coriSig((i-1)*snap+1:i*snap,:)).';
    R = X*X'/double(snap);
    [Pmusic, angle] = music_grid(R, array_num, target_num, grid_step);
    [~, peak_indices] = findpeaks(Pmusic, 'SortStr', 'descend', 'NPeaks', 2);
    theta_peaks = sort(angle(peak_indices));
    MUSIC_theta(i,:) = theta_peaks;
%     THETA = Rmusic(R, array_num, target_num);
%     disp(THETA)
%     MUSIC_theta(i,:) = round([THETA(1);THETA(3)]);
end


% 角度真值，从小到大排序
actual_theta = sort(real_theta,2);

% 计算每个样本的绝对误差
errors = abs(actual_theta - MUSIC_theta);

% 计算均方误差 (MSE)
squared_errors = errors .^ 2;
mse = mean(squared_errors(:)); % 将所有样本的平方误差展平并计算平均值

% 计算均方根误差 (RMSE)
rmse = sqrt(mse);

% 打印 RMSE
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmse);


% 统计不同阈值的正确率
% 误差阈值列表
thresholds = [0, 2, 4];

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

figure(1)
plot(1:smaple_num, MUSIC_theta(:,1),'ro');
hold on;
plot(1:smaple_num, MUSIC_theta(:,2),'bo');
legend('target1','target2')
xlabel('样本数')
ylabel('角度')
title('MUSIC结果')
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

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
% load("C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_oriSig_Array_10_20_theta_-60_51_target_2_snap_100_-10dB.mat")
% 获取复数矩阵
LoriSig = complex(L_test_R,L_test_I);
HoriSig = complex(H_test_R,H_test_I);
% NoriSig = complex(predict_H_R,predict_H_I);
coriSig = HoriSig;
% 获取阵列的阵元数
N = size(coriSig,2);
target_num = 2;
real_theta = test_angles;
% 记录每个样本的角度，两个目标
% tra_theta = zeros(length(one_hot_encoding),target_num);
smaple_num = size(real_theta, 1);
total_num_samples = size(coriSig, 1);
% 扫描范围
scan_theta = -60:1:60;
grid_step = 0.5;
array_num = N;

MUSIC_theta = zeros(smaple_num,target_num);

for i = 1:smaple_num
    X = squeeze(coriSig((i-1)*snap+1:i*snap,:)).';
    R = X*X'/double(snap);
    [Pmusic, angle] = music_grid(R, array_num, target_num, grid_step);
    [~, peak_indices] = findpeaks(Pmusic, 'SortStr', 'descend', 'NPeaks', 2);
    theta_peaks = sort(angle(peak_indices));
    MUSIC_theta(i,:) = theta_peaks;
%     THETA = Rmusic(R, array_num, target_num);
%     disp(THETA)
%     MUSIC_theta(i,:) = round([THETA(1);THETA(3)]);
end


% 角度真值，从小到大排序
actual_theta = sort(real_theta,2);

% 计算每个样本的绝对误差
errors = abs(actual_theta - MUSIC_theta);

% 计算均方误差 (MSE)
squared_errors = errors .^ 2;
mse = mean(squared_errors(:)); % 将所有样本的平方误差展平并计算平均值

% 计算均方根误差 (RMSE)
rmse = sqrt(mse);

% 打印 RMSE
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmse);


% 统计不同阈值的正确率
% 误差阈值列表
thresholds = [0, 2, 4];

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

% for i = 1 : length(one_hot_encoding)
%     disp(['样本数：',num2str(i)]);
%     % 获取一次样本数据
%     X = covSig(i,:).';
%     % 角度真值，从小到大排序
%     actual_theta = sort(real_theta(i,:));
%
%     % FFT测角，单快拍的情况
%     FFT_VALUE = zeros(181,1);
%     for theta = 1:length(scan_theta)
%         % 考虑阵列流形矢量
%         A = exp(-1i * 2 * pi * (0:N-1)' * 0.5 * sind(scan_theta(theta)));
%         % 记录每个角度导向矢量相关的值
%         FFT_VALUE(theta) = abs(A'*X);
%     end
%     % 找到 B_CBF 中最大的两个值及其索引
%     [sorted_values, sorted_indices] = sort(FFT_VALUE, 'descend');
%     % 根据目标数目选择最大的索引值
%     max_indices = sorted_indices(1:target_num);
%     % 获取对应的 THETA 值
%     max_theta_values = scan_theta(max_indices);
%     % 存储到矩阵中，从小到大排序
%     tra_theta(i,:) = sort(max_theta_values);
%
%
%
%
% %         计算协方差矩阵
% % R = (X * X') / snap; % 协方差矩阵
% %
% % % 使用CBF进行测试，数据是否有误
% % [B_CBF,THETA] = CBF(R,N,1);
% % % 找到 B_CBF 中最大的两个值及其索引
% % [sorted_values, sorted_indices] = sort(B_CBF, 'descend');
% % max_indices = sorted_indices(1:2);
%
% % figure(2);
% % plot(THETA,B_CBF,'LineWidth',1.5);
% % % hold on;
% % line([actual_doas(1), actual_doas(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1.2); % 红色虚线
% %     line([actual_doas(2), actual_doas(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1.2); % 红色虚线
% %
% % % hold off;
% % title("训练后预测CBF");
%
% % %             MUSIC验证
% % [B_MUSIC,THETA] = music_grid(R,N,target_num,0.5);
% % figure(2);
% % plot(THETA,B_MUSIC,'LineWidth',1.5);
% % line([sampled_angles(1,i), sampled_angles(1,i)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1.5); % 红色虚线
% % line([sampled_angles(2,i), sampled_angles(2,i)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1.5); % 红色虚线
% % xlabel('Angle/°');
% % ylabel('MUSIC Value/dB');
% % title("MUSIC");
% end
%
% % 统计结果
% % 假设 angle_values 和 FFT_theta 已经定义
% % angle_values 是真实的角度结果，形状为 (B, 2)
% % FFT_theta 是预测的角度结果，形状为 (B, 2)
% % % 存储到矩阵中，从小到大排序
% tra_theta = sort(tra_theta,2);
%
% % 角度真值，从小到大排序
% actual_theta = sort(real_theta,2);
%
% % threshold 是给定的误差阈值，例如 1.0 度
% threshold = 1.0;
%
% % 计算每个样本的绝对误差
% errors = abs(actual_theta - tra_theta);
%
% % 统计误差在阈值内的样本数量
% correct_predictions = sum(all(errors <= threshold, 2));
%
% % 计算准确率
% accuracy = correct_predictions / size(actual_theta, 1);

% % 打印结果
% fprintf('Number of correct predictions within threshold: %d\n', correct_predictions);
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);
%
% % 计算均方误差 (MSE)
% squared_errors = errors .^ 2;
% mse = mean(squared_errors(:)); % 将所有样本的平方误差展平并计算平均值
%
% % 计算均方根误差 (RMSE)
% rmse = sqrt(mse);
%
% % 打印 RMSE
% fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmse);
