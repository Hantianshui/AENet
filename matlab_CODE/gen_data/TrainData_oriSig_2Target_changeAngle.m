%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0728整理
% 脚本功能：产生双目标不同信噪比下的阵列接收信号，可以设置不同的角度间隔
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc
clear;
close all;
% 设置随机数种子
rng(25); % 训练数据设置随机数种子为25，可以选择任何你喜欢的数字


% 参数定义
snap = 1; % 采样点数，计算协方差矩阵用的快拍数
% SNR = -20:10:10; % 信噪比（假设为常数）
SNR = -10; % 信噪比（假设为常数）
theta_range = 10:1:30; % 角度范围
delta_theta = 10; % 两个信号源相隔角度
% theta = -25; % 设置目标角度
d = 0.5;

target = 2;

% 生成两个信号源的角度
source_angles = [theta_range; theta_range + delta_theta]; % 两个信号源的角度
% source_angles = randi([1, length(theta_range)], 1, num_samples);
% source_angles = theta_range + (0:target-1)'*delta_theta;
% source_angles = source_angles;
num_sources = size(source_angles, 1);

% 随机采样 20000 次，作为训练数据
total_num_samples = 100000;
% sampled_angles = datasample(source_angles, total_num_samples, 2, 'Replace', true);
% 随机抽取两个不重复的角度，重复抽取 20000 次
for i = 1:total_num_samples
    % 从范围内随机选择两个不同的角度
    random_angles = randsample(theta_range, 2, false);
    % 将这两个角度放入角度矩阵中的对应列
    sampled_angles(:, i) = random_angles;
end
% sampled_angles = randi([1, length(theta_range)], 2, total_num_samples);
% 阵列尺寸设置
array_sizes = [10, 15]; % 两种不同尺寸的阵列


for SNRindex = 1:length(SNR)
    % 获取每个循环的SNR
    snr_temp = SNR(SNRindex);

    % 数据存储
    % 分别表示小尺寸和大尺寸阵列的协方差矩阵
    L_train = zeros(total_num_samples,array_sizes(1),snap);
    H_train = zeros(total_num_samples,array_sizes(2),snap);

    % 生成数据，保证大小尺寸阵列的信号是相同的，只有导向矢量不同
    for t = 1:total_num_samples

        disp("样本数为:"+num2str(t))
        S = randn(num_sources, snap) + 1i * randn(num_sources, snap); % 复数信号，实部和虚部都是独立的随机值

        for a = 1:length(array_sizes)

            N = array_sizes(a); % 当前阵列尺寸
            disp("阵列尺寸为:"+num2str(N))

            % 生成阵列流形矩阵
            %             A = zeros(N, num_sources); % 存储阵列流形矩阵
%             A = exp(-1i * 2 * pi * (0:N-1)' * d * sind(source_angles')); % 考虑阵列流形矢量
            A = exp(-1i * 2 * pi * (0:N-1)' * d * sind(sampled_angles(:,t)')); % 考虑阵列流形矢量
            % 生成接收信号和计算协方差矩阵
            %         X = zeros(N, L);
            X = A*S;

            % 添加噪声，计算信号的功率
            

%             % 计算协方差矩阵
%             R = (X * X') / snap; % 协方差矩阵
% 
%             % 使用CBF进行测试，数据是否有误
%             [B_CBF,THETA] = CBF(R,N,0.5);
%             figure(1);
%             plot(THETA,B_CBF,'LineWidth',1.5);
%             % hold on;
%             line([sampled_angles(1,t), sampled_angles(1,t)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             line([sampled_angles(2,t), sampled_angles(2,t)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             % hold off;
%             title("训练后预测CBF");
% 
%             %             MUSIC验证
%             [B_MUSIC,THETA] = music_grid(R,N,num_sources,0.5);
%             figure(2);
%             plot(THETA,B_MUSIC,'LineWidth',1.5);
%             line([sampled_angles(1,t), sampled_angles(1,t)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             line([sampled_angles(2,t), sampled_angles(2,t)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             xlabel('Angle/°');
%             ylabel('MUSIC Value/dB');
%             title("MUSIC");

            % 根据情况分开存储
            if a==1
                X = awgn(X,snr_temp,'measured');
                L_train(t,:,:) = X;
            end
            if a==2
                H_train(t,:,:) = X;
            end

        end
    end

    % 将复数拆分为实部和虚部，所有的数据，需要划分训练集和验证集
    L_train_R_all = real(L_train);
    L_train_I_all = imag(L_train);
    H_train_R_all = real(H_train);
    H_train_I_all = imag(H_train);

    train_data_size = total_num_samples*0.9; % 0.8的数据用来做训练
    val_data_size = total_num_samples*0.1; % 0.2的数据用来做验证

    split_point = round(total_num_samples - val_data_size);

    % 随机划分
    seq = randperm(total_num_samples);

    % 存储到变量中
    L_train_R = L_train_R_all(seq(1:split_point),:,:);
    L_train_I = L_train_I_all(seq(1:split_point),:,:);
    H_train_R = H_train_R_all(seq(1:split_point),:,:);
    H_train_I = H_train_I_all(seq(1:split_point),:,:);
    train_angles = sampled_angles(:,seq(1:split_point))';
%     train_angles = source_angles;

    L_val_R = L_train_R_all(seq(split_point+1:end),:,:);
    L_val_I = L_train_I_all(seq(split_point+1:end),:,:);
    H_val_R = H_train_R_all(seq(split_point+1:end),:,:);
    H_val_I = H_train_I_all(seq(split_point+1:end),:,:);
    val_angles = sampled_angles(:,seq(split_point+1:end))';
%     val_angles = source_angles;

    % 保存数据

    % 定义新文件夹的名称
    folderName = 'TrainData_oriSig_2Target_changeAngle';

    % 检查文件夹是否已经存在
    if exist(folderName, 'dir')
        disp('文件夹已经存在，无需创建。');
    else
        % 创建新文件夹
        mkdir(folderName);
        disp('新文件夹已创建。');
    end

    filename=['Train_oriSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
        '_angle_' num2str(source_angles(1,1)) '_' num2str(source_angles(1,end)) ...
        '_snap_' num2str(snap) '_' num2str(snr_temp) 'dB' '.mat'];

%     filename=['Train_covSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))... 
%         '_Theta_' num2str(theta_range(1)) '_' num2str(theta_range(end))...
%         '_DeltaTheta' num2str(delta_theta) '_snap' num2str(snap) '_' num2str(SNR) 'dB' '.mat'];
    disp(['文件名为：' filename]);

    save(fullfile(folderName, filename), 'L_train_R', 'L_train_I', 'H_train_R', 'H_train_I','train_angles',...
        'L_val_R', 'L_val_I', 'H_val_R', 'H_val_I','val_angles','snr_temp','snap','-v7.3');
    disp(['SNR为' num2str(snr_temp) '保存完成'])
end
