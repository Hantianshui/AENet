%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0728整理
% 脚本功能：产生双目标不同信噪比、不同角度间隔下的阵列接收信号，可以设置不同的角度间隔
% 训练集包含所有的信噪比条件和角度间隔条件，角度是固定间隔后随机抽取的
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc;
clear;
close all;

% 设置随机数种子
rng(25); % 训练数据设置随机数种子为25，可以选择任何你喜欢的数字

% 参数定义
snap = 1; % 采样点数，计算协方差矩阵用的快拍数
SNR = 0:10:30; % 信噪比（假设为常数）
d = 0.5;
% 阵列尺寸设置

target_num = 2;
num_sources = 2;

% 定义角度范围和步长
theta_start = -60;      % 第一个目标的起始角度
theta_end = 60;         % 最大角度
theta_diffs = 1:1:14; % 不同的角度间隔

% 每种组合产生的样本数量
num_samples_per_combination = 20;

% 总样本数量
total_num_samples = 50000;  % 你可以根据需要调整总样本数量

% 阵列尺寸设置
array_sizes = [10, 20]; % 两种不同尺寸的阵列

% 定义新文件夹的名称
folderName = 'TrainData_oriSig_2Target_all';

% 检查文件夹是否已经存在
if exist(folderName, 'dir')
    disp('文件夹已经存在，无需创建。');
else
    % 创建新文件夹
    mkdir(folderName);
    disp('新文件夹已创建。');
end

% 初始化HDF5文件
filename = fullfile(folderName, 'TrainData_oriSig_2Target_all.h5');
if exist(filename, 'file')
    delete(filename);
end

for theta_diff = theta_diffs
    % 生成第一个目标的角度范围
    theta_range = [theta_start, theta_end - theta_diff];

    % 初始化 real_theta
    real_theta = zeros(total_num_samples, 2);

    % 随机抽取第一个角度并计算第二个角度
    for i = 1:total_num_samples
        theta1 = theta_range(1) + (theta_range(2) - theta_range(1)) * rand;
        theta2 = theta1 + theta_diff;
        real_theta(i, :) = [theta1, theta2];
    end

    for SNRindex = 1:length(SNR)
        % 获取每个循环的SNR
        snr_temp = SNR(SNRindex);

        % 数据存储
        % 分别表示小尺寸和大尺寸阵列的协方差矩阵
        L_train = zeros(total_num_samples, array_sizes(1), snap);
        H_train = zeros(total_num_samples, array_sizes(2), snap);

        % 生成数据，每次信号数据相同，这样不同的只有导向矢量，和论文中一致
        for t = 1:total_num_samples

            disp("样本数为:" + num2str(t))
            S = randn(num_sources, snap) + 1i * randn(num_sources, snap); % 复数信号，实部和虚部都是独立的随机值

            for a = 1:length(array_sizes)

                N = array_sizes(a); % 当前阵列尺寸
                disp("阵列尺寸为:" + num2str(N))

                % 生成阵列流形矩阵
                A = exp(-1i * 2 * pi * (0:N-1)' * d * sind(real_theta(t, :))); % 考虑阵列流形矢量

                X = A * S;

                % 添加噪声，计算信号的功率
                X = awgn(X, snr_temp, 'measured');

                % 根据情况分开存储，保存原始的信号
                if a == 1
                    L_train(t, :, :) = X;
                end
                if a == 2
                    H_train(t, :, :) = X;
                end

            end
        end

        % 将复数拆分为实部和虚部，所有的数据，需要划分训练集和验证集
        L_train_R = real(L_train);
        L_train_I = imag(L_train);
        H_train_R = real(H_train);
        H_train_I = imag(H_train);

        train_angles = real_theta;

        % 存储数据到HDF5文件
        group_name = ['/SNR_' num2str(snr_temp) '_thetaDiff_' num2str(theta_diff)];
        h5create(filename, [group_name '/L_train_R'], size(L_train_R));
        h5write(filename, [group_name '/L_train_R'], L_train_R);

        h5create(filename, [group_name '/L_train_I'], size(L_train_I));
        h5write(filename, [group_name '/L_train_I'], L_train_I);

        h5create(filename, [group_name '/H_train_R'], size(H_train_R));
        h5write(filename, [group_name '/H_train_R'], H_train_R);

        h5create(filename, [group_name '/H_train_I'], size(H_train_I));
        h5write(filename, [group_name '/H_train_I'], H_train_I);

        h5create(filename, [group_name '/train_angles'], size(train_angles));
        h5write(filename, [group_name '/train_angles'], train_angles);

        disp(['SNR为' num2str(snr_temp) '，角度间隔为' num2str(theta_diff) '的数据保存到HDF5文件中'])
    end
end

disp(['所有数据已保存到文件' filename])
