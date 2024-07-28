clc;
close all;
clear all;
%% 导入数据
SNR = -10;
flag_all = 0;
% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNR\Test_Array_10_20_target_2_10dB.mat')
load(['C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeSNR\Test_oriSig_Array_10_20_target_2_',num2str(SNR),'dB.mat'])
Htrue_data_complex = complex(H_test_R, H_test_I);
Ltrue_data_complex = complex(L_test_R, L_test_I);

% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_Array_10_20_target_2_10dB.mat')
load(['C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\CNN_Predict_Test_oriSig_Array_10_20_target_2_',num2str(SNR),'dB.mat'])
complex_data_CNN = complex(predict_H_R, predict_H_I);


% load('C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_Array_10_20_target_2_10dB.mat')
load(['C:\Users\buaa\Desktop\毕设相关\孔径扩展\python_pytorch\DNN_Predict_Test_oriSig_Array_10_20_target_2_',num2str(SNR),'dB.mat'])
complex_data_DNN = complex(predict_H_R, predict_H_I);

snap = double(snap);
%% 绘图
% 初始化一个数组来存储每个数据集的CBF值
theta_grid = 0.5;
x_line = -90:theta_grid:90;
num_samples = length(Htrue_data_complex); % 假设所有数据集的长度相同
B_CBF_DNN = zeros(num_samples, length(x_line));
B_CBF_CNN = zeros(num_samples, length(x_line));
B_CBF_Htrue = zeros(num_samples, length(x_line));
B_CBF_Ltrue = zeros(num_samples, length(x_line));

if(flag_all==0)
    for i = 1:length(Htrue_data_complex)
        % for i = 1:1000

        disp(i);
        M = size(complex_data_DNN,2);
        X = squeeze(complex_data_DNN(i,:,:));
        X = X*X'/snap;
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
        X = X*X'/snap;
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
        X = X*X'/snap;
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
        X = X*X'/snap;
        [B_CBF,THETA] = CBF(X,M,0.5);
        B_CBF_Ltrue(i,:) = B_CBF;

        figure(4);
        plot(THETA,B_CBF,'LineWidth',1.5);
        line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
        line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
        xlabel('Angle/°');
        ylabel('CBF Value/dB');
        title("LtrueCBF");



        figure(5)
        plot(THETA,B_CBF_Ltrue(i,:),'LineWidth',1.5);
        hold on;
        plot(THETA,B_CBF_Htrue(i,:),'LineWidth',1.5);
        plot(THETA,B_CBF_CNN(i,:),'LineWidth',1.5);
        plot(THETA,B_CBF_DNN(i,:),'LineWidth',1.5);
        line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
        line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
        legend("LtrueCBF","HtrueCBF","CNNCBF","DNNCBF","真值1","真值2");
        xlabel('Angle/°');
        ylabel('CBF Value/dB');
        title("四种情况CBF对比图")
        % 绘制红色虚线，不分配图例标签
        hold off;
    end
else
    for i = 1:length(Htrue_data_complex)
        % for i = 1:1000

        disp(i);
        M = size(complex_data_DNN,2);
        X = squeeze(complex_data_DNN(i,:,:));
        X = X*X'/snap;
        [B_CBF,THETA] = CBF(X,M,0.5);
        B_CBF_DNN(i,:) = B_CBF;

        M = size(complex_data_CNN,2);
        X = squeeze(complex_data_CNN(i,:,:));
        X = X*X'/snap;
        [B_CBF,THETA] = CBF(X,M,0.5);
        B_CBF_CNN(i,:) = B_CBF;

        M = size(Htrue_data_complex,2);
        X = squeeze(Htrue_data_complex(i,:,:));
        X = X*X'/snap;
        [B_CBF,THETA] = CBF(X,M,0.5);
        B_CBF_Htrue(i,:) = B_CBF;

        M = size(Ltrue_data_complex,2);
        X = squeeze(Ltrue_data_complex(i,:,:));
        X = X*X'/snap;
        [B_CBF,THETA] = CBF(X,M,0.5);
        B_CBF_Ltrue(i,:) = B_CBF;


    end
    figure(1)
    plot(THETA,mean(B_CBF_Ltrue,1),'LineWidth',1.5);
    hold on;
    plot(THETA,mean(B_CBF_Htrue,1),'LineWidth',1.5);
    plot(THETA,mean(B_CBF_CNN,1),'LineWidth',1.5);
    plot(THETA,mean(B_CBF_DNN,1),'LineWidth',1.5);
    line([test_angles(1), test_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    line([test_angles(2), test_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--','LineWidth',1); % 红色虚线
    legend("LtrueCBF","HtrueCBF","CNNCBF","DNNCBF","真值1","真值2");
    xlabel('Angle/°');
    ylabel('CBF Value/dB');
    title(sprintf("%d组数据平均四种情况CBF对比图", length(Htrue_data_complex)))
    % 绘制红色虚线，不分配图例标签
    hold off;
    % 保存图形
    filename = sprintf('CBF_ALL_oriSig_Array_10_20_target_2_%ddB.png', SNR);
    saveas(gcf, filename);
end







