%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0728整理
% 脚本功能：产生单目标不同信噪比下，角度区间内的原始阵列接收信号
% 每一个文件是一个测试集数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% clc
clear;
close all;
% 设置随机数种子
rng(45); % 训练数据设置随机数种子为25，可以选择任何你喜欢的数字


% 参数定义
snap = 1; % 采样点数，计算协方差矩阵用的快拍数
% SNR = -20:2:10; % 信噪比（假设为常数）
SNR = 30; % 信噪比（假设为常数）
% theta_range = 0:0.5:45; % 角度范围
% theta_range = 25; % 角度范围
% delta_theta = 10; % 两个信号源相隔角度
% theta = 5; % 设置目标角度
d = 0.5;

target = 1;
num_sources = 1;
% 定义角度范围
theta_range = -60:1:60;

% 每个角度产生10个样本
num_samples_per_combination = 100;
total_num_samples = length(theta_range) * num_samples_per_combination;


angle_values = zeros(total_num_samples, 1);
% 为每个组合生成样本的 one-hot 编码和保存角度值
sample_idx = 1;
for i = 1:length(theta_range)
    % 获取当前组合的角度
    angles = theta_range(i);
    
    % 在 one-hot 编码矩阵中设置对应的位置为 1
    for j = 1:num_samples_per_combination
        angle_values(sample_idx) = angles;  % 保存角度值
        sample_idx = sample_idx + 1;
    end
end


% 阵列尺寸设置
array_sizes = [10, 20]; % 两种不同尺寸的阵列


for SNRindex = 1:length(SNR)
    % 获取每个循环的SNR
    snr_temp = SNR(SNRindex);

    % 数据存储
    % 分别表示小尺寸和大尺寸阵列的协方差矩阵
    L_test = zeros(total_num_samples,array_sizes(1),snap);
    H_test = zeros(total_num_samples,array_sizes(2),snap);

    % 生成数据，保证信号在大小尺寸阵列中一致
    for t = 1:total_num_samples

        disp("样本数为:"+num2str(t))
        S = randn(num_sources, snap) + 1i * randn(num_sources, snap); % 复数信号，实部和虚部都是独立的随机值

        for a = 1:length(array_sizes)

            N = array_sizes(a); % 当前阵列尺寸
            disp("阵列尺寸为:"+num2str(N))

            % 生成阵列流形矩阵
            A = zeros(N, num_sources); % 存储阵列流形矩阵
            A(:, 1) = exp(-1i * 2 * pi * (0:N-1)' * d * sind(angle_values(t))); % 考虑阵列流形矢量

            % 生成接收信号和计算协方差矩阵
            %         X = zeros(N, L);
            X = A*S;

            % 添加噪声，计算信号的功率
            X = awgn(X,snr_temp,'measured');

            % 计算协方差矩阵
%             R = (X * X') / snap; % 协方差矩阵

            %             % 使用CBF进行测试，数据是否有误
            %             [B_CBF,THETA] = CBF(R,N,0.5);
            %             figure(1);
            %
            %             plot(THETA,B_CBF,'LineWidth',1.5);
            %             % hold on;
            %             line([source_angles, source_angles], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
            % %             line([sampled_angles(2,i), sampled_angles(2,i)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
            %             % hold off;
            %             title("训练后预测CBF");


            % 根据情况分开存储
            if a==1
                L_test(t,:,:) = X;
            end
            if a==2
                H_test(t,:,:) = X;
            end

        end
    end

    % 将复数拆分为实部和虚部，所有的数据，需要划分训练集和验证集
    L_test_R = real(L_test);
    L_test_I = imag(L_test);
    H_test_R = real(H_test);
    H_test_I = imag(H_test);


    %     train_angles = sampled_angles(:,seq(1:split_point))';
    test_angles = angle_values;


    % 保存数据

    % 定义新文件夹的名称
    folderName = 'TestData_oriSig_single_changeSNR';

    % 检查文件夹是否已经存在
    if exist(folderName, 'dir')
        disp('文件夹已经存在，无需创建。');
    else
        % 创建新文件夹
        mkdir(folderName);
        disp('新文件夹已创建。');
    end

    filename=['Test_oriSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
        '_target_' num2str(num_sources) '_' num2str(snr_temp) 'dB' '.mat'];

    disp(['文件名为：' filename]);

    save(fullfile(folderName, filename), 'L_test_R', 'L_test_I', 'H_test_R', 'H_test_I',...
        'test_angles','snr_temp','snap','-v7.3');
    disp(['SNR为' num2str(snr_temp) '保存完成'])


end