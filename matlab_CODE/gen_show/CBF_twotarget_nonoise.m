clc;
close all;
clear all;
%% 导入数据
% load("TrainData_changeSNR\Train_Array_10_20_target_1_8dB.mat")
% load("TrainData_2Target_changeSNR\Train_Array_10_20_target_2_-10dB.mat")

load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNAP\Test_Array_10_20_target_2_snap_10_999dB.mat')
Htrue_nonoise_data_complex = complex(H_test_R, H_test_I);
% Ltrue_nonoise_data_complex = complex(L_test_R, L_test_I);

% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNR\Test_Array_10_20_target_2_10dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNAP\Test_Array_10_20_target_2_snap_10_-10dB.mat')
Htrue_data_complex = complex(H_test_R, H_test_I);
Ltrue_data_complex = complex(L_test_R, L_test_I);

% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_Array_10_20_target_2_10dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_Array_10_20_target_2_snap_10_999dB.mat')
complex_data_CNN = complex(predict_H_R, predict_H_I);


% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_Array_10_20_target_2_10dB.mat')
load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_Array_10_20_target_2_snap_10_999dB.mat')
complex_data_DNN = complex(predict_H_R, predict_H_I);

%% 绘图



% 初始化一个数组来存储每个数据集的CBF值
theta_grid = 0.5;
x_line = -90:theta_grid:90;
num_samples = length(Htrue_data_complex); % 假设所有数据集的长度相同
B_CBF_DNN = zeros(num_samples, length(x_line));
B_CBF_CNN = zeros(num_samples, length(x_line));
B_CBF_Htrue = zeros(num_samples, length(x_line));
B_CBF_Ltrue = zeros(num_samples, length(x_line));
B_CBF_Htrue_nonoise = zeros(num_samples, length(x_line));
% B_CBF_Ltrue = zeros(num_samples, length(x_line));


for i = 1:length(Htrue_data_complex)
    % for i = 1:1000
    disp(i);
    M = size(complex_data_DNN,2);
    X = squeeze(complex_data_DNN(i,:,:));
    [B_CBF,THETA] = CBF(X,M,0.5);
    B_CBF_DNN(i,:) = B_CBF;

    figure(1);
    plot(THETA,B_CBF,'LineWidth',1.5);
    %     hold on;
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    % hold off;
    xlabel('Angle/°');
    ylabel('CBF Value/dB');
    title("DNN训练后预测CBF");

    M = size(complex_data_CNN,2);
    X = squeeze(complex_data_CNN(i,:,:));
    [B_CBF,THETA] = CBF(X,M,0.5);
    B_CBF_CNN(i,:) = B_CBF;

    figure(2);
    plot(THETA,B_CBF,'LineWidth',1.5);
    % hold on;
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    % hold off;
    xlabel('Angle/°');
    ylabel('CBF Value/dB');
    title("CNN训练后预测CBF");


    M = size(Htrue_data_complex,2);
    X = squeeze(Htrue_data_complex(i,:,:));
    [B_CBF,THETA] = CBF(X,M,0.5);
    B_CBF_Htrue(i,:) = B_CBF;

    figure(3);
    plot(THETA,B_CBF,'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    xlabel('Angle/°');
    ylabel('CBF Value/dB');
    title("HtrueCBF");



    M = size(Ltrue_data_complex,2);
    X = squeeze(Ltrue_data_complex(i,:,:));
    [B_CBF,THETA] = CBF(X,M,0.5);
    B_CBF_Ltrue(i,:) = B_CBF;

    figure(4);
    plot(THETA,B_CBF,'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    xlabel('Angle/°');
    ylabel('CBF Value/dB');
    title("LtrueCBF");

    M = size(Htrue_nonoise_data_complex,2);
    X = squeeze(Htrue_nonoise_data_complex(i,:,:));
    [B_CBF,THETA] = CBF(X,M,0.5);
    B_CBF_Htrue_nonoise(i,:) = B_CBF;

    figure(5);
    plot(THETA,B_CBF,'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    xlabel('Angle/°');
    ylabel('CBF Value/dB');
    title("Htrue-nonoise");


    figure(6)
%     plot(THETA,B_CBF_Ltrue(i,:),'LineWidth',1.5);
    plot(THETA,B_CBF_Htrue(i,:),'LineWidth',1.5);
    hold on;
    
    plot(THETA,B_CBF_Htrue_nonoise(i,:),'LineWidth',1.5);
    plot(THETA,B_CBF_CNN(i,:),'LineWidth',1.5);
    plot(THETA,B_CBF_DNN(i,:),'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
%     legend("LtrueCBF","HtrueCBF","HtrueCBF-nonoise","CNNCBF","DNNCBF","真值1","真值2");
    legend("HtrueCBF","HtrueCBF-nonoise","CNNCBF","DNNCBF","真值1","真值2");
    xlabel('Angle/°');
    ylabel('CBF Value/dB');
    title("四种情况CBF对比图")
    % 绘制红色虚线，不分配图例标签
    hold off;


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

figure(1)
plot(THETA,mean(B_CBF_Htrue,1),'LineWidth',1.5);
hold on;
plot(THETA,mean(B_CBF_Htrue_nonoise,1),'LineWidth',1.5);
plot(THETA,mean(B_CBF_CNN,1),'LineWidth',1.5);
plot(THETA,mean(B_CBF_DNN,1),'LineWidth',1.5);
line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
legend("HtrueCBF","HtrueCBF-nonoise","CNNCBF","DNNCBF","真值1","真值2");
xlabel('Angle/°');
ylabel('CBF Value/dB');
title("10000组数据平均四种情况CBF对比图")
% 绘制红色虚线，不分配图例标签
hold off;




% 定义一个函数来绘制CBF和相应的测试角度线
function plotCBF(X, THETA, test_angle, titleStr)
figure;
subplot(2, 2, 1);
plot(THETA, X, 'LineWidth', 1.5);
hold on;
% line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1.5); % 红色虚线
% line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1.5); % 红色虚线
title(titleStr);
xlabel('Angle');
ylabel('CBF Value');
hold off;
end





