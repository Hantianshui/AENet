%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0728整理
% 脚本功能：产生双目标不同信噪比下的阵列接收信号，可以设置不同的角度间隔
% 每个文件是一个固定信噪比固定角度间隔的测试集，其中角度也是随机抽取的
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc
clear;
close all;
% 设置随机数种子
rng(45); % 训练数据设置随机数种子为25，可以选择任何你喜欢的数字


% 参数定义
snap = 100; % 采样点数，计算协方差矩阵用的快拍数
% SNR = -20:10:10; % 信噪比（假设为常数）
SNR = 999; % 信噪比（假设为常数）
% theta_range = 10:0.5:35; % 角度范围
theta_range = 10:1:30; % 角度范围
delta_theta = 10; % 两个信号源相隔角度
% theta = -25; % 设置目标角度
d = 0.5;

target = 2;

% 生成两个信号源的角度
source_angles = [theta_range; theta_range + delta_theta]; % 两个信号源的角度
% source_angles = theta_range + (0:target-1)'*delta_theta;
num_sources = size(source_angles, 1);

% 随机采样 20000 次，作为训练数据
total_num_samples = 1000;
sampled_angles = datasample(source_angles, total_num_samples, 2, 'Replace', true);
% % 随机抽取两个不重复的角度，重复抽取 20000 次
% for i = 1:total_num_samples
%     % 从范围内随机选择两个不同的角度
%     random_angles = randsample(theta_range, 2, false);
%     % 将这两个角度放入角度矩阵中的对应列
%     sampled_angles(:, i) = random_angles;
% end
% 阵列尺寸设置
array_sizes = [10, 15]; % 两种不同尺寸的阵列


for SNRindex = 1:length(SNR)
    % 获取每个循环的SNR
    snr_temp = SNR(SNRindex);

    % 数据存储
    % 分别表示小尺寸和大尺寸阵列的协方差矩阵
    L_test = zeros(total_num_samples*snap,array_sizes(1));
    H_test = zeros(total_num_samples*snap,array_sizes(2));

    % 生成数据，每次信号数据相同，这样不同的只有导向矢量，和论文中一致
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
%             X = awgn(X,snr_temp,'measured');

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
%                 L_test(t,:,:) = X;
                L_test((t-1)*snap + 1 : t*snap,:) = X.';
            end
            if a==2
%                 H_test(t,:,:) = X;
                H_test((t-1)*snap + 1 : t*snap,:) = X.';
            end

        end
    end

    % 将复数拆分为实部和虚部，所有的数据，需要划分训练集和验证集
    L_test_R = real(L_test);
    L_test_I = imag(L_test);
    H_test_R = real(H_test);
    H_test_I = imag(H_test);

    test_angles = sampled_angles;
    % 保存数据

    % 定义新文件夹的名称
    %     folderName = 'TestData_2Target_changeSNR';
    folderName = 'TestData_oriSig_2Target_changeAngle';

    % 检查文件夹是否已经存在
    if exist(folderName, 'dir')
        disp('文件夹已经存在，无需创建。');
    else
        % 创建新文件夹
        mkdir(folderName);
        disp('新文件夹已创建。');
    end


%     filename=['Test_covSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
%         '_angle_' num2str(source_angles(1,1)) '_' num2str(source_angles(1,end)) '_' num2str(snr_temp) 'dB' '.mat'];
%     filename=['Test_covSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))... 
%         '_angle_' num2str(theta_range(1)) '_' num2str(theta_range(end))...
%         '_DeltaTheta' num2str(delta_theta) '_snap' num2str(snap) '_' num2str(SNR) 'dB' '.mat'];
    filename=['Test_oriSig' '_Array_' num2str(array_sizes(1)) '_' num2str(array_sizes(2))...
        '_angle_' num2str(source_angles(1,1)) '_' num2str(source_angles(1,end)) ...
        '_snap_' num2str(snap) '_' num2str(snr_temp) 'dB' '.mat'];
    disp(['文件名为：' filename]);

    save(fullfile(folderName, filename), 'L_test_R', 'L_test_I', 'H_test_R', 'H_test_I',...
        'test_angles','snr_temp','snap','-v7.3');
    disp(['SNR为' num2str(snr_temp) '保存完成'])
end
