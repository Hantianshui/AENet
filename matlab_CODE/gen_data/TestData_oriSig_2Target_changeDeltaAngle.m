%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0728整理
% 脚本功能：产生双目标不同信噪比、不同角度间隔下的阵列接收信号，可以设置不同的角度间隔
% 每一个文件是一个测试集数据，包含所有的信噪比条件和角度间隔条件，角度是固定间隔后随机抽取的
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc;
clear;
close all;

% 设置随机数种子
rng(45);

% 参数定义
snap = 1; % 采样点数，计算协方差矩阵用的快拍数
SNR = 20; % 信噪比（假设为常数）
d = 0.5;
num_sources = 2;

% 阵列尺寸设置
array_sizes = [10, 20]; % 两种不同尺寸的阵列

% 设置角度范围
theta_start = -60;   % 角度起始值
theta_end = 60;     % 角度结束值

% 设置每种角度差的样本数
num_samples_per_diff = 5000;

% 初始化存储数组
total_num_samples = num_samples_per_diff * 12;
real_theta = zeros(total_num_samples, 2);
L_test = zeros(total_num_samples, array_sizes(1), snap);
H_test = zeros(total_num_samples, array_sizes(2), snap);
labels = zeros(total_num_samples, 1); % 用于存储每个样本表示的间隔

% 数据生成和存储
sample_index = 1;

for theta_diff = 1:12
    for trial = 1:num_samples_per_diff
        % 随机生成第一个角度
        random_theta1 = (theta_end - theta_start) * rand() + theta_start;
        random_theta2 = random_theta1 + theta_diff;

        % 确保角度在范围内
        while random_theta2 > theta_end
            random_theta1 = (theta_end - theta_start) * rand() + theta_start;
            random_theta2 = random_theta1 + theta_diff;
        end

        real_theta(sample_index, :) = [random_theta1, random_theta2];
        labels(sample_index) = theta_diff;

        S = randn(num_sources, snap) + 1i * randn(num_sources, snap); % 复数信号，实部和虚部都是独立的随机值

        for a = 1:length(array_sizes)
            N = array_sizes(a); % 当前阵列尺寸

            % 生成阵列流形矩阵
            A = exp(-1i * 2 * pi * (0:N-1)' * d * sind(real_theta(sample_index, :))); % 考虑阵列流形矢量

            % 生成接收信号和计算协方差矩阵
            X = A * S;

            % 添加噪声
            X = awgn(X, SNR, 'measured');

            % 根据情况分开存储
            if a == 1
                L_test(sample_index, :, :) = X;
            end
            if a == 2
                H_test(sample_index, :, :) = X;
            end
        end

        sample_index = sample_index + 1;
    end
end

% 将复数拆分为实部和虚部
L_test_R = real(L_test);
L_test_I = imag(L_test);
H_test_R = real(H_test);
H_test_I = imag(H_test);

% 定义新文件夹的名称
folderName = 'TestData_oriSig_2Target_changeSNR';

% 检查文件夹是否已经存在
if exist(folderName, 'dir')
    disp('文件夹已经存在，无需创建。');
else
    % 创建新文件夹
    mkdir(folderName);
    disp('新文件夹已创建。');
end

% 保存数据到一个文件中
filename = ['Test_oriSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
    '_theta_' num2str(theta_start) '_' num2str(theta_end)...
    '_target_' num2str(num_sources) '_' num2str(SNR) 'dB' '.mat'];

disp(['文件名为：' filename]);

save(fullfile(folderName, filename), 'L_test_R', 'L_test_I', 'H_test_R', 'H_test_I', 'real_theta', 'labels', 'SNR', 'snap', '-v7.3');
disp('所有样本保存完成');
