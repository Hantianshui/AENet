import os

import hdf5storage
import joblib
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

# from CovmatrixCNNNet import CovmatrixCNN
# from OriSigCNNNet import OriSigCNN
from OriSigCNNNet import OriSigCNN
# 数据加载器
class MyDataset(Dataset):
    def __init__(self, L):
        self.L = L

    def __len__(self):
        return len(self.L)

    def __getitem__(self, idx):
        Test_data = torch.tensor(self.L[idx],dtype=torch.float32)
        return Test_data

# 超参数
# 导入模型和归一化工具的文件夹
# dirStore = f"Train_Array_10_20_target_2_-20dB_CNN_2024-05-20-10-33-25/"
dirStore = f"Train_oriSig_Array_10_20_theta_-60_52_target_2_30dB_CNN_2024-06-30-15-04-00/"

# dirStore = f"./" + file_name + "_CNN_" + nowTime
# 读入数据
# v = hdf5storage.loadmat(r"C:\Users\buaa\Desktop\毕设相关\阵列扩展\matlab部分\TestData\Test_Array_10_20_Theta_0_45_DeltaTheta10_snap10_0dB.mat")
# v = hdf5storage.loadmat(r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_changeSNR\Test_Array_10_20_target_1_-10dB.mat")
# 获取文件路径
# 单一目标
# file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_single_changeSNR\Test_oriSig_Array_10_20_target_1_30dB.mat"
# 多目标改变SNR
file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeSNR\Test_oriSig_Array_10_20_theta_-60_52_target_2_30dB.mat"
# file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeSNR\Test_oriSig_Array_10_20_theta_-60_52_target_2_snap_100_30dB.mat"
# 多目标改变快拍数
# file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_2Target_changeSNAP\Test_Array_10_20_target_2_snap_10_999dB.mat"
# 多目标改变角度
# file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TestData_oriSig_2Target_changeAngle\Test_oriSig_Array_10_15_angle_10_30_snap_100_-10dB.mat"
# 读入数据并进行预处理
v = hdf5storage.loadmat(file_path)

# 获取文件名
file_name_with_extension = os.path.basename(file_path)
file_name, file_extension = os.path.splitext(file_name_with_extension)

print("用于区分的file_name:",file_name)  # 输出文件名（无后缀）：Train_Array_10_20_Theta_0_45_10dB

snap = int(v['snap'].item())
ArrayL = 10
ArrayH = 20
# SNR = int(v['snr_temp'].item())
# SNR = int(v['SNR'].item())
# snap = int(v['snap'].item())
# DeltaTheta = int(v['delta_theta'].item())
# Theta_start = int(v['theta_start'].item())
# Theta_end = int(v['theta_end'].item())

# 测试集数据
test_angles = v['test_angles']
# test_angles = v['real_theta']
test_L_R = v['L_test_R']
test_L_I = v['L_test_I']
# labels = v['labels']
print("处理前的维度_train_L_I:",test_L_R.shape)
print("处理前的维度_val_L_R:",test_L_I.shape)

# 按照num*channel*H*W的格式进行转换

# test_L = np.concatenate((test_L_R[:, np.newaxis, :, :], test_L_I[:, np.newaxis, :, :]), axis=1)

# test_L_R_flat = test_L_R.transpose((0, 2, 1)).reshape(-1, 1,test_L_R.shape[1])
# test_L_I_flat = test_L_I.transpose((0, 2, 1)).reshape(-1, 1,test_L_I.shape[1])
#
# test_L = np.concatenate((test_L_R_flat, test_L_I_flat), axis=1)
test_L = np.stack((test_L_R, test_L_I), axis=1)
# 打印 test_L 和 test_H 的形状
print("合并双通道后的维度_train_L_I:",test_L.shape)  # 输出：(16000, 200)

H_L = test_L.shape[2]
H_H = 20
test_L_reshaped = test_L.reshape(test_L.shape[0], -1)
# 读入归一化器进行测试数据归一化和反归一化
# filename = f"Train_Array_scalarL_Theta_{Theta0_train}_{Theta1_train}_{SNR}dB_CNN"
filename = dirStore + f"Train_Array_scalarL_CNN"
mm_scalerRT_INPUT_L = joblib.load(filename)

# filename = f"Train_Array_scalarH_Theta_{Theta0_train}_{Theta1_train}_{SNR}dB_CNN"
filename = dirStore + f"Train_Array_scalarH_CNN"
mm_scalerRT_INPUT_H = joblib.load(filename)

# test_L_reshaped = test_L.reshape(test_L.shape[0], -1)
test_L = mm_scalerRT_INPUT_L.transform(test_L_reshaped)
test_L = test_L.reshape(test_L.shape[0],2,-1)
# # 将数据reshape成一个二维数组，因为归一化对矩阵作用的话，是对每一列进行的
# test_L = test_L.reshape(-1, ArrayL*snap*2)
#
# # Normalization，分别对两个尺寸的阵列进行标准化，用训练集进行fit，测试集只进行transform
# test_L = mm_scalerRT_INPUT_L.transform(test_L)
#
# # 将数据反变换reshape成一个二维数组，因为归一化对矩阵作用的话，是对每一列进行的
# test_L = test_L.reshape(-1, 2, ArrayL, snap)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_files_with_extension(directory, extension):
    # 检查后缀是否以'.'开头，如果没有则添加
    if not extension.startswith('.'):
        extension = '.' + extension

    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory) if f.endswith(extension)]

    return files


# 示例用法
directory_path = dirStore  # 替换为你的文件夹路径
extension = 'pth'  # 替换为你想要查找的文件后缀

filename = dirStore + get_files_with_extension(directory_path, extension)[0]

# 导入保存的模型参数值
# filename = f"Train_Array_{ArrayL}_{ArrayH}_Theta_{Theta0_train}_{Theta1_train}_{SNR}dB_CNN.h5"
# filenameme = dirStore + f"Train_Array_10_20_Theta_0_45_10dB_CNN.pth"

# 测试集进行测试
# 实例化模型并将其移动到 GPU
# H_L = 10
# H_H = 20
model = OriSigCNN(H_L,2*H_H,2)


# 加载模型的状态字典
model.load_state_dict(torch.load(filename))

# 可视化网络
# dummy_input = torch.randn(10, 2, 10)  # 根据你的模型输入尺寸调整
# import torch
# import torch.onnx
# # 导出模型为 ONNX 格式
# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=['input'], output_names=['output'])


model.to(device)
# 创建训练集和测试集的 Dataset
test_dataset = MyDataset(test_L)

# L_R, L_I, H_R, H_I = train_dataset[0]
# print(train_dataset[0])
# print("{:.16f}".format(L_R[0,0]))
# 创建 DataLoader
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model.eval()  # 设置模型为评估模式

# 在推断时遍历数据加载器，进行预测
predictions = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        predictions.append(outputs)

# 将所有预测结果拼接为一个张量
predictions = torch.cat(predictions, dim=0)
# 需要先移动到CPU然后转为numpy进行后续操作
predict_H_all = predictions.cpu().numpy()
predict_H_all = predict_H_all.reshape(predict_H_all.shape[0],-1)
predict_H_all = mm_scalerRT_INPUT_H.inverse_transform(predict_H_all)
predict_H_all = predict_H_all.reshape(predict_H_all.shape[0],2,-1)
# 将拼接后的数据 train_L 按照第二维度分割成两个矩阵
# predict_H_R_flat, predict_H_I_flat = np.split(predict_H_all, 2, axis=1)
predict_H_R_flat = predict_H_all[:, 0, :]  # 第一个通道
predict_H_I_flat = predict_H_all[:, 1, :]  # 第二个通道
# 将每个矩阵重新变形为原始形状
predict_H_R = predict_H_R_flat.reshape(test_L_R.shape[0], H_H)
predict_H_I = predict_H_I_flat.reshape(test_L_R.shape[0], H_H)


# # Denormalization
# # 反归一化
# # 将数据reshape成一个二维数组，因为归一化对矩阵作用的话，是对每一列进行的
# prdict_H_all = prdict_H_all.reshape(-1, ArrayH*snap*2)
#
# predict_H_all = mm_scalerRT_INPUT_H.inverse_transform(prdict_H_all)
# # 将数据反变换reshape成一个二维数组，因为归一化对矩阵作用的话，是对每一列进行的
# predict_H_all = predict_H_all.reshape(-1, 2, ArrayH, snap)
#
# # 沿着第二个维度拆分数组
# predict_H_R = predict_H_all[:, 0]  # 实部部分
# predict_H_I = predict_H_all[:, 1]  # 虚部部分


# 打印结果数组的形状
print("predict_H_R shape:", predict_H_R.shape)  # (10000, 20, 20)
print("predict_H_I shape:", predict_H_I.shape)  # (10000, 20, 20)


filename = f"CNN_Predict_" + file_name + ".mat"
sio.savemat(filename, {'predict_H_I': predict_H_I, 'predict_H_R': predict_H_R, 'test_angles': test_angles, 'snap':snap})
print(filename + "保存成功！凸(艹皿艹 )")





