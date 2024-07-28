clc;
close all;
clear all;
%% 导入数据
% load("TrainData_changeSNR\Train_Array_10_20_target_1_8dB.mat")
% load("TrainData_2Target_changeSNR\Train_Array_10_20_target_2_-10dB.mat")
% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNR\Test_Array_10_20_target_2_-20dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNAP\Test_Array_10_20_target_2_snap_10_-10dB.mat')
Htrue_data_complex = complex(H_test_R, H_test_I);
Ltrue_data_complex = complex(L_test_R, L_test_I);

% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_Array_10_20_target_2_-20dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_Array_10_20_target_2_snap_10_-10dB.mat')
complex_data_CNN = complex(predict_H_R, predict_H_I);


% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_Array_10_20_target_2_-20dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_Array_10_20_target_2_snap_10_-10dB.mat')
complex_data_DNN = complex(predict_H_R, predict_H_I);

%% 绘图



% 初始化一个数组来存储每个数据集的CBF值
theta_grid = 0.25;
x_line = -90:theta_grid:90;

num_samples = length(Htrue_data_complex); % 假设所有数据集的长度相同
B_CBF_DNN = zeros(num_samples, length(x_line));
B_CBF_CNN = zeros(num_samples, length(x_line));
B_CBF_Htrue = zeros(num_samples, length(x_line));
B_CBF_Ltrue = zeros(num_samples, length(x_line));
theta_CBF_DNN = zeros(num_samples, 2);
theta_CBF_CNN = zeros(num_samples, 2);
theta_CBF_Htrue = zeros(num_samples, 2);
theta_CBF_Ltrue = zeros(num_samples, 2);


for i = 1:length(Htrue_data_complex)
    % for i = 1:1000
    disp(i);
    M = size(complex_data_DNN,2);
    X = squeeze(complex_data_DNN(i,:,:));
    [B_CBF,THETA] = CBF(X,M,theta_grid);
    B_CBF_DNN(i,:) = B_CBF;


    % 查找峰值的位置和幅度
    [peaks, peak_locs] = findpeaks(B_CBF);

    % 找到幅度最高的两个峰值的位置
    [sorted_peaks, sorted_indices] = sort(peaks, 'descend');
    top_peak_locs = peak_locs(sorted_indices(1:2));
    theta_CBF_DNN(i,:) = sort(THETA(top_peak_locs));

%     figure(1);
%     plot(THETA,B_CBF,'LineWidth',1.5);
%     %     hold on;
%     line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     % hold off;
%     xlabel('Angle/°');
%     ylabel('CBF Value/dB');
%     title("DNN训练后预测CBF");


    M = size(complex_data_CNN,2);
    X = squeeze(complex_data_CNN(i,:,:));
    [B_CBF,THETA] = CBF(X,M,theta_grid);
    B_CBF_CNN(i,:) = B_CBF;

    % 查找峰值的位置和幅度
    [peaks, peak_locs] = findpeaks(B_CBF);

    % 找到幅度最高的两个峰值的位置
    [sorted_peaks, sorted_indices] = sort(peaks, 'descend');
    top_peak_locs = peak_locs(sorted_indices(1:2));
    theta_CBF_CNN(i,:) = sort(THETA(top_peak_locs));

%     figure(2);
%     plot(THETA,B_CBF,'LineWidth',1.5);
%     % hold on;
%     line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     % hold off;
%     xlabel('Angle/°');
%     ylabel('CBF Value/dB');
%     title("CNN训练后预测CBF");


    M = size(Htrue_data_complex,2);
    X = squeeze(Htrue_data_complex(i,:,:));
    [B_CBF,THETA] = CBF(X,M,theta_grid);
    B_CBF_Htrue(i,:) = B_CBF;

    % 查找峰值的位置和幅度
    [peaks, peak_locs] = findpeaks(B_CBF);

    % 找到幅度最高的两个峰值的位置
    [sorted_peaks, sorted_indices] = sort(peaks, 'descend');
    top_peak_locs = peak_locs(sorted_indices(1:2));
    theta_CBF_Htrue(i,:) = sort(THETA(top_peak_locs));

%     figure(3);
%     plot(THETA,B_CBF,'LineWidth',1.5);
%     line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     xlabel('Angle/°');
%     ylabel('CBF Value/dB');
%     title("HtrueCBF");


    M = size(Ltrue_data_complex,2);
    X = squeeze(Ltrue_data_complex(i,:,:));
    [B_CBF,THETA] = CBF(X,M,theta_grid);
    B_CBF_Ltrue(i,:) = B_CBF;

    % 查找峰值的位置和幅度
    [peaks, peak_locs] = findpeaks(B_CBF);

    % 找到幅度最高的两个峰值的位置
    [sorted_peaks, sorted_indices] = sort(peaks, 'descend');
    top_peak_locs = peak_locs(sorted_indices(1:2));
    theta_CBF_Ltrue(i,:) = sort(THETA(top_peak_locs));

%     figure(4);
%     plot(THETA,B_CBF,'LineWidth',1.5);
%     line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     xlabel('Angle/°');
%     ylabel('CBF Value/dB');
%     title("LtrueCBF");


%     figure(5)
%     plot(THETA,B_CBF_Ltrue(i,:),'LineWidth',1.5);
%     hold on;
%     plot(THETA,B_CBF_Htrue(i,:),'LineWidth',1.5);
%     plot(THETA,B_CBF_CNN(i,:),'LineWidth',1.5);
%     plot(THETA,B_CBF_DNN(i,:),'LineWidth',1.5);
%     line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     legend("LtrueCBF","HtrueCBF","CNNCBF","DNNCBF","真值1","真值2");
%     xlabel('Angle/°');
%     ylabel('CBF Value/dB');
%     title("四种情况CBF对比图")
%     % 绘制红色虚线，不分配图例标签
%     hold off;


end

% figure(1)
% plot(THETA,mean(B_CBF_Ltrue,1),'LineWidth',1.5);
% hold on;
% plot(THETA,mean(B_CBF_Htrue,1),'LineWidth',1.5);
% plot(THETA,mean(B_CBF_CNN,1),'LineWidth',1.5);
% plot(THETA,mean(B_CBF_DNN,1),'LineWidth',1.5);
% line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
% line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
% legend("LtrueCBF","HtrueCBF","CNNCBF","DNNCBF","真值1","真值2");
% xlabel('Angle/°');
% ylabel('CBF Value/dB');
% title("10000组数据平均四种情况CBF对比图")
% % 绘制红色虚线，不分配图例标签
% hold off;


% 计算MSE
MSE_CNN = calMSE(theta_CBF_CNN, test_angles);
MSE_DNN = calMSE(theta_CBF_DNN, test_angles);
MSE_Htrue = calMSE(theta_CBF_Htrue, test_angles);
MSE_Ltrue = calMSE(theta_CBF_Ltrue, test_angles);
% 显示结果
disp(['MSE_CNN for data: ', num2str(MSE_CNN)]);
% 显示结果
disp(['MSE_DNN for data: ', num2str(MSE_DNN)]);
% 显示结果
disp(['MSE_Htrue for data: ', num2str(MSE_Htrue)]);
% 显示结果
disp(['MSE_Ltrue for data: ', num2str(MSE_Ltrue)]);







