%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0728整理
% 脚本功能：产生双目标不同信噪比下，固定角度间隔，角度区间内的原始阵列接收信号
% 每一个文件是一个测试集数据，包含一个信噪比条件下和一个角度间隔条件下的原始阵列接收信号数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc
clear;
close all;
% 设置随机数种子
rng(45); % 训练数据设置随机数种子为25，可以选择任何你喜欢的数字


% 参数定义
snap = 1; % 采样点数，计算协方差矩阵用的快拍数
% SNR = -10:10:30; % 信噪比（假设为常数）
SNR = 20; % 信噪比（假设为常数）
% theta_range = -40:1:40; % 角度范围
d = 0.5;
% 阵列尺寸设置

target_num = 2;
num_sources = 2;


% 固定角度差抽取多个角度对
% 定义角度范围和步长
theta_start = -60;   % 角度起始值
theta_end = 60;       % 角度结束值
theta_diff = 4;       % 角度差

% 设置抽取次数
num_trials = 50000;

real_theta = zeros(num_trials,2);


for trial = 1:num_trials
    % 随机生成第一个角度
    random_theta1 = (theta_end - theta_start) * rand() + theta_start;

    % 计算第二个角度
    random_theta2 = random_theta1 + theta_diff;
    real_theta(trial,:) = [random_theta1;random_theta2];
    fprintf('第 %d 次抽取的角度对：\n', trial);
    fprintf('theta1 = %.1f°\n', random_theta1);
    fprintf('theta2 = %.1f°\n', random_theta2);
    fprintf('角度差为 %.1f°\n\n', random_theta2 - random_theta1);
end


% % 定义角度范围
% theta_range = -60:1:60;
% 
% % 找出所有不重复的两角度组合
% combinations = nchoosek(theta_range, 2);
% 
% 
% % 每种组合产生10个样本
% num_samples_per_combination = 1;
% total_num_samples = size(combinations, 1) * num_samples_per_combination;
% 
% % 初始化 one-hot 编码矩阵和角度存储数组
% one_hot_encoding = zeros(total_num_samples, length(theta_range));
% 
% % 为每个组合生成样本的 one-hot 编码和保存角度值
% [combs_idx, reps] = ndgrid(1:size(combinations, 1), 1:num_samples_per_combination);
% combs_idx = combs_idx(:);
% real_theta = combinations(combs_idx, :);
% 


% % 定义角度范围和步长
% theta_start = -60;   % 第一个目标的起始角度
% theta_step = 1;        % 角度步长
% theta_end = 60;        % 最大角度
% theta_diff = 8;        % 第二个目标比第一个目标大 8°
% 
% % 生成第一个目标的角度
% theta1 = theta_start:theta_step:(theta_end - theta_diff);
% 
% % 生成第二个目标的角度
% theta2 = theta1 + theta_diff;
% 
% % 确保角度不超过指定范围
% valid_indices = theta2 <= theta_end;
% theta1 = theta1(valid_indices);
% theta2 = theta2(valid_indices);
% 
% % 组合成两角度对
% combinations = [theta1' theta2'];
% theta_range = [theta1(1), theta1(end)];
% 
% % 每种组合产生100个样本
% num_samples_per_combination = 1;
% num_combinations = size(combinations, 1);
% 
% % 初始化 real_theta
% real_theta = zeros(num_combinations * num_samples_per_combination, 2);
% 
% % 生成每种组合的样本
% for i = 1:num_combinations
%     start_idx = (i-1) * num_samples_per_combination + 1;
%     end_idx = i * num_samples_per_combination;
%     real_theta(start_idx:end_idx, :) = repmat(combinations(i, :), num_samples_per_combination, 1);
% end

% 显示结果
% disp(real_theta);
% 

total_num_samples = num_trials


% 定义角度范围和步长
% theta_start = -60;      % 第一个目标的起始角度
% theta_end = 60;         % 最大角度
% theta_diff = 8;         % 第二个目标比第一个目标大 8°
% 
% 生成第一个目标的角度范围
theta_range = [theta_start, theta_end - theta_diff];
% 
% 每种组合产生的样本数量
% num_samples_per_combination = 20;
% 
% 总样本数量
% total_num_samples = 5000;  % 你可以根据需要调整总样本数量
% 
% 初始化 real_theta
% real_theta = zeros(total_num_samples, 2);
% 
% 随机抽取第一个角度并计算第二个角度
% for i = 1:total_num_samples
%     theta1 = theta_range(1) + (theta_range(2) - theta_range(1)) * rand;
%     theta2 = theta1 + theta_diff;
%     real_theta(i, :) = [theta1, theta2];
% end


% 阵列尺寸设置
array_sizes = [10, 20]; % 两种不同尺寸的阵列


% real_theta = [20,30]




for SNRindex = 1:length(SNR)
    % 获取每个循环的SNR
    snr_temp = SNR(SNRindex);

    % 数据存储
    % 分别表示小尺寸和大尺寸阵列的协方差矩阵
    L_test = zeros(total_num_samples,array_sizes(1),snap);
    H_test = zeros(total_num_samples,array_sizes(2),snap);

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

            % 生成接收信号和计算协方差矩阵
            X = A*S;
%             X0 = X;

            % 添加噪声，计算信号的功率
            X = awgn(X,snr_temp,'measured');

%             % 计算协方差矩阵
%             R = (X * X') / snap; % 协方差矩阵
% 
% %             使用CBF进行测试，数据是否有误
%             [B_CBF,THETA] = CBF(R,N,0.5);
%             figure(1);
% 
%             plot(THETA,B_CBF,'LineWidth',1.5);
%             hold on;
%             line([source_angles(1), source_angles(1)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             line([source_angles(2), source_angles(2)], ylim, 'Color', 'r', 'LineStyle', '--'); % 红色虚线
%             hold off;
%             title("CBF");
% 
%             %             MUSIC验证
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

    test_angles = real_theta;
    % 保存数据

    % 定义新文件夹的名称
    %     folderName = 'TestData_2Target_changeSNR';
    folderName = 'TestData_oriSig_2Target_changeSNR';

    % 检查文件夹是否已经存在
    if exist(folderName, 'dir')
        disp('文件夹已经存在，无需创建。');
    else
        % 创建新文件夹
        mkdir(folderName);
        disp('新文件夹已创建。');
    end

    %     filename=['Test' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
    %         '_target_' num2str(num_sources) '_' num2str(snr_temp) 'dB' '.mat'];
    filename=['Test_oriSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
        '_theta_' num2str(theta_range(1)) '_' num2str(theta_range(end))...
        '_target_' num2str(num_sources) '_' num2str(snr_temp) 'dB' '.mat'];

    disp(['文件名为：' filename]);

    save(fullfile(folderName, filename), 'L_test_R', 'L_test_I', 'H_test_R', 'H_test_I',...
        'test_angles','snr_temp','snap','-v7.3');
    disp(['SNR为' num2str(snr_temp) '保存完成'])
end
