clc;
close all;
clear all;
%% 导入数据
% load("TrainData_changeSNR\Train_Array_10_20_target_1_8dB.mat")
% load("TrainData_2Target_changeSNR\Train_Array_10_20_target_2_-10dB.mat")

% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNR\Test_Array_10_20_target_2_10dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_kron_2Target_changeSNR\Test_oriSig_kron_Array_10_20_target_2_10dB.mat')
Htrue_data_complex = complex(H_test_R, H_test_I);
Ltrue_data_complex = complex(L_test_R, L_test_I);

% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_Array_10_20_target_2_10dB.mat')
% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_oriSig_Array_10_20_target_2_10dB.mat')
% complex_data_CNN = complex(predict_H_R, predict_H_I);


% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_Array_10_20_target_2_10dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_oriSig_kron_Array_10_20_target_2_10dB.mat')
complex_data_DNN = complex(predict_H_R, predict_H_I);

target_num = 2;
snap = double(snap);
%% 绘图
% 初始化一个数组来存储每个数据集的CBF值
theta_grid = 0.5;
x_line = -90:theta_grid:90;
num_samples = length(Htrue_data_complex); % 假设所有数据集的长度相同
B_MUSIC_DNN = zeros(num_samples, length(x_line));
B_MUSIC_CNN = zeros(num_samples, length(x_line));
B_MUSIC_Htrue = zeros(num_samples, length(x_line));
B_MUSIC_Ltrue = zeros(num_samples, length(x_line));

for i = 1:length(Htrue_data_complex)/snap

    disp(i);
    
    M = size(complex_data_DNN,2);
    X = squeeze(complex_data_DNN(i,:,:));
%     X = squeeze(complex_data_DNN((i-1)*snap+1:i*snap,:))';
    X = X*X'/snap;
    [B_MUSIC,THETA] = music_kron_grid(X,M,target_num,0.5);
    B_MUSIC_DNN(i,:) = B_MUSIC;

    figure(1);
    plot(THETA,B_MUSIC,'LineWidth',1.5);
    %     hold on;
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    % hold off;
    xlabel('Angle/°');
    ylabel('MUSIC Value/dB');
    title("DNN训练后预测MUSIC");


%     M = size(complex_data_CNN,2);
%     X = squeeze(complex_data_CNN(i,:,:));
%     X = X*X'/snap;
%     [B_MUSIC,THETA] = music_grid(X,M,target_num,0.5);
%     B_MUSIC_CNN(i,:) = B_MUSIC;
% 
%     figure(2);
%     plot(THETA,B_MUSIC,'LineWidth',1.5);
%     % hold on;
%     line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     % hold off;
%     xlabel('Angle/°');
%     ylabel('MUSIC Value/dB');
%     title("CNN训练后预测MUSIC");



    M = size(Htrue_data_complex,2);
    X = squeeze(Htrue_data_complex(i,:,:));
%     X = squeeze(Htrue_data_complex((i-1)*snap+1:i*snap,:))';
    X = X*X'/snap;
    [B_MUSIC,THETA] = music_kron_grid(X,M,target_num,0.5);
    B_MUSIC_Htrue(i,:) = B_MUSIC;

    figure(3);
    plot(THETA,B_MUSIC,'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    xlabel('Angle/°');
    ylabel('MUSIC Value/dB');
    title("HtrueMUSIC");



    M = size(Ltrue_data_complex,2);
    X = squeeze(Ltrue_data_complex(i,:,:));
%     X = squeeze(Ltrue_data_complex((i-1)*snap+1:i*snap,:))';
    X = X*X'/snap;
    [B_MUSIC,THETA] = music_kron_grid(X,M,target_num,0.5);
    B_MUSIC_Ltrue(i,:) = B_MUSIC;

    figure(4);
    plot(THETA,B_MUSIC,'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    xlabel('Angle/°');
    ylabel('MUSIC Value/dB');
    title("LtrueMUSIC");



    figure(5)
    plot(THETA,B_MUSIC_Ltrue(i,:),'LineWidth',1.5);
    hold on;
    plot(THETA,B_MUSIC_Htrue(i,:),'LineWidth',1.5);
    plot(THETA,B_MUSIC_CNN(i,:),'LineWidth',1.5);
    plot(THETA,B_MUSIC_DNN(i,:),'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    legend("LtrueMUSIC","HtrueMUSIC","CNNMUSIC","DNNMUSIC","真值1","真值2");
    xlabel('Angle/°');
    ylabel('MUSIC Value/dB');
    title("四种情况MUSIC对比图")
    % 绘制红色虚线，不分配图例标签
    hold off;
 end

figure(1)
plot(THETA,mean(B_MUSIC_Ltrue,1),'LineWidth',1.5);
hold on;
plot(THETA,mean(B_MUSIC_Htrue,1),'LineWidth',1.5);
plot(THETA,mean(B_MUSIC_CNN,1),'LineWidth',1.5);
plot(THETA,mean(B_MUSIC_DNN,1),'LineWidth',1.5);
line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
legend("LtrueMUSIC","HtrueMUSIC","CNNMUSIC","DNNMUSIC","真值1","真值2");
xlabel('Angle/°');
ylabel('MUSIC Value/dB');
title("10000组数据平均四种情况MUSIC对比图")
% 绘制红色虚线，不分配图例标签
hold off;








