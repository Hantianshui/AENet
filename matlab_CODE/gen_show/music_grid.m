function [Pmusic, angle] = music_grid(R, array_num, target_num,grid_step)
% 获取协方差矩阵
Rxx = R;

%% 特征值分解
[EV,D]=eig(Rxx);                   %特征值分解
EVA=diag(D)';                      %将特征值矩阵对角线提取并转为一行
[~,I]=sort(EVA);                 %将特征值排序 从小到大
EV=fliplr(EV(:, I));                % 对应特征矢量排序
En = EV(:, target_num+1 : array_num);  % 取矩阵的第M+1到N列组成噪声子空间

% 设置阵列信息，生成阵元的位置
dd = 0.5;
d = 0 : dd : (array_num-1)*dd;

% 设置角度遍历网格值
angle = -90:grid_step:90;
angle_num = length(angle);

%% 根据网格搜索
% 遍历每个角度，计算空间谱
for index = 1:angle_num
    phim = pi/180*angle(index);
    a = exp(-1i*2*pi*d*sin(phim)).';     
    Pmusic(index)=1/(a'*En*En'*a);
end

Pmusic = abs(Pmusic);
Pmmax = max(Pmusic);
Pmusic = 10*log10(Pmusic/Pmmax);            % 归一化处理dB的格式