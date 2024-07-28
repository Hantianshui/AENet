% function [output1, output2, ...] = functionName(input1, input2, ...)

function [Theta, spectrum]  = music(R,k,N)
% MUSIC算法
[U, D] = eig(R); % R的特征向量和特征值
[~, idx] = sort(diag(D), 'descend'); % 按降序排列特征值
noise_subspace = U(:, idx(k+1:end)); % 噪声子空间
Theta = -90:1:90;
spectrum = zeros(length(Theta), 1); % 存储MUSIC谱
for i = 1:length(Theta)
    theta_scan = Theta(i);
    steering_vector = exp(-1i * 2 * pi *0.5* (0:N-1)' * sind(theta_scan)); % 扫描角度的阵列流形矢量
    spectrum(i) = 1 / (steering_vector' * noise_subspace * noise_subspace' * steering_vector); % MUSIC谱计算
end
spectrum = abs(spectrum);
Pmmax=max(spectrum);
spectrum=10*log10(spectrum/Pmmax); 
