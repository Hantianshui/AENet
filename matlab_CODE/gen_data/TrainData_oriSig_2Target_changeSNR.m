%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0728整理
% 脚本功能：产生双目标不同信噪比下的阵列接收信号，可以设置不同的角度间隔
% 每个文件是一个固定信噪比固定角度间隔的训练集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc
clear;
close all;
% 设置随机数种子
rng(25); % 训练数据设置随机数种子为25，可以选择任何你喜欢的数字


% 参数定义
snap = 1; % 采样点数，计算协方差矩阵用的快拍数
SNR = 0:10:30; % 信噪比（假设为常数）
% SNR = 0; % 信噪比（假设为常数）
% theta_range = -40:1:40; % 角度范围
d = 0.5;
% 阵列尺寸设置

% 目标个数设置
target_num = 2;
num_sources = 2;

% 定义角度范围和步长
theta_start = -60;      % 第一个目标的起始角度
theta_end = 60;         % 最大角度
theta_diff = 8;         % 第二个目标比第一个目标大 8°

% 生成第一个目标的角度范围
theta_range = [theta_start, theta_end - theta_diff];

% 每种组合产生的样本数量
num_samples_per_combination = 20;

% 总样本数量
total_num_samples = 80000;  % 你可以根据需要调整总样本数量

% 初始化 real_theta
real_theta = zeros(total_num_samples, 2);

% 随机抽取第一个角度并计算第二个角度
for i = 1:total_num_samples
    theta1 = theta_range(1) + (theta_range(2) - theta_range(1)) * rand;
    theta2 = theta1 + theta_diff;
    real_theta(i, :) = [theta1, theta2];
end

% 显示生成的角度对
% disp(real_theta);


% 显示结果
% disp(real_theta);
% 

% total_num_samples = num_combinations * num_samples_per_combination;

% 阵列尺寸设置
array_sizes = [10, 20]; % 两种不同尺寸的阵列


for SNRindex = 1:length(SNR)
    % 获取每个循环的SNR
    snr_temp = SNR(SNRindex);

    % 数据存储
    % 分别表示小尺寸和大尺寸阵列的协方差矩阵
    L_train = zeros(total_num_samples,array_sizes(1),snap);
    H_train = zeros(total_num_samples,array_sizes(2),snap);

    % 生成数据，每次信号数据相同，这样不同的只有导向矢量，和论文中一致

    for t = 1:total_num_samples

        disp("样本数为:"+num2str(t))
        S = randn(num_sources, snap) + 1i * randn(num_sources, snap); % 复数信号，实部和虚部都是独立的随机值

        for a = 1:length(array_sizes)

            N = array_sizes(a); % 当前阵列尺寸
            disp("阵列尺寸为:"+num2str(N))

            % 生成阵列流形矩阵
            %             A = zeros(N, num_sources); % 存储阵列流形矩阵
            A = exp(-1i * 2 * pi * (0:N-1)' * d * sind(real_theta(t,:))); % 考虑阵列流形矢量


            %             S = awgn(S,snr_temp,'measured');
            X = A*S;
%             noise = randn(N, snap) + 1i * randn(N, snap);
%         
%             % 计算信号功率
%             signal_power = mean(mean(abs(X).^2));
%         
%             % 计算噪声功率
%             noise_power = mean(mean(abs(noise).^2));
%         
%             % 计算信噪比（SNR）
%             snr = 10 * log10(signal_power / noise_power);
%         
%             % 显示信噪比
%             disp("信噪比为:" + num2str(snr) + " dB")

%             添加噪声，计算信号的功率
            X = awgn(X,snr_temp,'measured');

            % 计算协方差矩阵
%             R = (X * X') / snap; % 协方差矩阵
    
% %             使用CBF进行测试，数据是否有误
%             [B_CBF,THETA] = CBF(R,N,0.5);
%             figure(1);
%             plot(THETA,B_CBF,'LineWidth',1.5);
%             % hold on;
%             line([source_angles(1), source_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             line([source_angles(2), source_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             % hold off;
%             xlabel('Angle/°');
%             ylabel('CBF Value/dB');
%             title("CBF");
% 
% %             MUSIC验证
%             [B_MUSIC,THETA] = music_grid(R,N,num_sources,0.5);
%             figure(2);
%             plot(THETA,B_MUSIC,'LineWidth',1.5);
%             line([source_angles(1), source_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             line([source_angles(2), source_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             xlabel('Angle/°');
%             ylabel('MUSIC Value/dB');
%             title("MUSIC");

% % 传统方法扫描范围
% scan_theta = -60:1:60;
% 
% % 定义导向矢量矩阵
% steering_matrix = zeros(N, length(scan_theta));
% for idx = 1:length(scan_theta)
%     theta = scan_theta(idx);
%     steering_matrix(:, idx) = exp(1i * 2 * pi * (0:N-1)' * 0.5 * sind((theta)));
% end
% R = X.';
%     B_CBF = abs(R*steering_matrix);
%     [~, peak_indices] = findpeaks(B_CBF, 'SortStr', 'descend', 'NPeaks', 2);
%     theta_peaks = scan_theta(peak_indices);
%     CBF_theta = sort(round(theta_peaks));




            % 根据情况分开存储，保存原始的信号
            if a==1
                L_train(t,:,:) = X;
            end
            if a==2
                H_train(t,:,:) = X;
            end

        end
    end

    % 将复数拆分为实部和虚部，所有的数据，需要划分训练集和验证集
    L_train_R = real(L_train);
    L_train_I = imag(L_train);
    H_train_R = real(H_train);
    H_train_I = imag(H_train);

    train_angles = real_theta;

%     train_data_size = total_num_samples*0.8; % 0.8的数据用来做训练
%     val_data_size = total_num_samples*0.2; % 0.2的数据用来做验证
% 
%     split_point = round(total_num_samples - val_data_size);
% 
%     % 随机划分
%     seq = randperm(total_num_samples);
% 
%     % 存储到变量中
%     L_train_R = L_train_R_all(seq(1:split_point),:,:);
%     L_train_I = L_train_I_all(seq(1:split_point),:,:);
%     H_train_R = H_train_R_all(seq(1:split_point),:,:);
%     H_train_I = H_train_I_all(seq(1:split_point),:,:);
%     %     train_angles = sampled_angles(:,seq(1:split_point))';
%     train_angles = source_angles;
% 
%     L_val_R = L_train_R_all(seq(split_point+1:end),:,:);
%     L_val_I = L_train_I_all(seq(split_point+1:end),:,:);
%     H_val_R = H_train_R_all(seq(split_point+1:end),:,:);
%     H_val_I = H_train_I_all(seq(split_point+1:end),:,:);
%     %     val_angles = sampled_angles(:,seq(split_point+1:end))';
%     val_angles = source_angles;

    % 保存数据

    % 定义新文件夹的名称
    folderName = 'TrainData_oriSig_2Target_changeSNR';

    % 检查文件夹是否已经存在
    if exist(folderName, 'dir')
        disp('文件夹已经存在，无需创建。');
    else
        % 创建新文件夹
        mkdir(folderName);
        disp('新文件夹已创建。');
    end

    filename=['Train_oriSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
        '_theta_' num2str(theta_range(1)) '_' num2str(theta_range(end))...
        '_target_' num2str(num_sources) '_' num2str(snr_temp) 'dB' '.mat'];

    disp(['文件名为：' filename]);

%     save(fullfile(folderName, filename), 'L_train_R', 'L_train_I', 'H_train_R', 'H_train_I','train_angles',...
%         'L_val_R', 'L_val_I', 'H_val_R', 'H_val_I','val_angles', ...
%         'snr_temp','snap','-v7.3');
    save(fullfile(folderName, filename), 'L_train_R', 'L_train_I', 'H_train_R', 'H_train_I','train_angles',...
        'snr_temp','snap','-v7.3');
    disp(['SNR为' num2str(snr_temp) '保存完成'])
end
