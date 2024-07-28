import os
from datetime import datetime

import hdf5storage
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# from CovmatrixCNNNet import CovmatrixCNN
from tqdm import tqdm
# from OriSigCNNNet import OriSigCNN
from OriSigCNNNet import OriSigCNN
class MyDataset(Dataset):
    def __init__(self, L, H):
        self.L = L
        self.H = H

    def __len__(self):
        return len(self.L)

    def __getitem__(self, idx):
        Train_data = torch.tensor(self.L[idx],dtype=torch.float32)
        Train_label = torch.tensor(self.H[idx],dtype=torch.float32)
        return Train_data, Train_label
# 初始化函数
def initialize_weights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
# 初始化函数
def initialize_weights_xavier(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
import h5py
import numpy as np


def load_hdf5_data(filename):
    L_R_list = []
    L_I_list = []
    H_R_list = []
    H_I_list = []

    with h5py.File(filename, 'r') as f:
        for group_name in f.keys():
            L_train_R = np.array(f[group_name + '/L_train_R']).T
            L_train_I = np.array(f[group_name + '/L_train_I']).T
            H_train_R = np.array(f[group_name + '/H_train_R']).T
            H_train_I = np.array(f[group_name + '/H_train_I']).T

            # 合并当前组的数据到列表中
            L_R_list.append(L_train_R)
            L_I_list.append(L_train_I)
            H_R_list.append(H_train_R)
            H_I_list.append(H_train_I)

    # 将所有组合的数据合并成大的集合
    L_R = np.vstack(L_R_list)
    L_I = np.vstack(L_I_list)
    H_R = np.vstack(H_R_list)
    H_I = np.vstack(H_I_list)

    # 将实部和虚部分别合并形成复数数据
    L = L_R + 1j * L_I
    H = H_R + 1j * H_I

    # 打乱数据
    indices = np.arange(L.shape[0])
    np.random.shuffle(indices)

    L = L[indices]
    H = H[indices]

    # 按9.5:0.5的比例拆分数据
    split_idx = int(0.95 * L.shape[0])

    train_L = L[:split_idx]
    val_L = L[split_idx:]
    train_H = H[:split_idx]
    val_H = H[split_idx:]

    # 将复数数据拆分为实部和虚部
    train_L_R = train_L.real
    train_L_I = train_L.imag
    train_H_R = train_H.real
    train_H_I = train_H.imag
    val_L_R = val_L.real
    val_L_I = val_L.imag
    val_H_R = val_H.real
    val_H_I = val_H.imag

    return train_L_R, train_L_I, train_H_R, train_H_I, val_L_R, val_L_I, val_H_R, val_H_I

if __name__ == '__main__':
    # 超参数
    num_epochs = 100
    batch_size = 500
    learningRate = 0.001
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取文件路径
    # file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TrainData_2Target_changeSNR\Train_Array_10_20_target_2_-20dB.mat"
    # file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TrainData_2Target_changeSNAP\Train_Array_10_20_target_2_snap_100_-10dB.mat"
    # file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TrainData_oriSig_2Target_changeSNR\Train_oriSig_Array_10_20_target_2_0dB.mat"
    # 多目标改变信噪比
    file_path_train = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TrainData_oriSig_2Target_changeSNR\Train_oriSig_Array_10_20_theta_-60_52_target_2_30dB.mat"
    file_path_val = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\ValData_oriSig_2Target_changeSNR\Val_oriSig_Array_10_20_theta_-60_52_target_2_30dB.mat"
    # # 多目标改变角度
    # # file_path = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TrainData_oriSig_2Target_changeAngle\Train_oriSig_Array_10_15_angle_10_30_snap_1_-10dB.mat"
    # # 单目标改变信噪比
    # # file_path_train = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\TrainData_oriSig_single_changeSNR\Train_oriSig_Array_10_20_target_1_5dB.mat"
    # # file_path_val = r"C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分\ValData_oriSig_single_changeSNR\Val_oriSig_Array_10_20_target_1_5dB.mat"
    # 读入数据并进行预处理
    v_train = hdf5storage.loadmat(file_path_train)
    v_val = hdf5storage.loadmat(file_path_val)
    # 获取文件名
    file_name_with_extension = os.path.basename(file_path_train)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    print("用于区分的file_name:",file_name)  # 输出文件名（无后缀）：Train_Array_10_20_Theta_0_45_10dB



    SNR = 20
    ArrayL = 10
    ArrayH = 20
    Theta0 = 0
    Theta1 = 45
    snap = (int)(v_train['snap'].item())

    # 提取数据
    train_L_R = v_train['L_train_R']
    train_L_I = v_train['L_train_I']
    train_H_R = v_train['H_train_R']
    train_H_I = v_train['H_train_I']
    val_L_R = v_val['L_val_R']
    val_L_I = v_val['L_val_I']
    val_H_R = v_val['H_val_R']
    val_H_I = v_val['H_val_I']


    #
    # filename = r'C:\Users\buaa\Desktop\毕设相关\孔径扩展\matlab部分/TrainData_oriSig_2Target_all/TrainData_oriSig_2Target_all.h5'
    # train_L_R, train_L_I, train_H_R, train_H_I, val_L_R, val_L_I, val_H_R, val_H_I = load_hdf5_data(filename)
    #
    # # 获取文件名和扩展名
    # file_name_with_extension = os.path.basename(filename)
    # file_name, file_extension = os.path.splitext(file_name_with_extension)
    #
    # # 输出文件名（无后缀）
    # print("用于区分的file_name:", file_name)


    # 训练集数据转换
    # train_L_R_flat = train_L_R.transpose((0, 2, 1)).reshape(-1, 1,train_L_R.shape[1])
    # train_L_I_flat = train_L_I.transpose((0, 2, 1)).reshape(-1, 1,train_L_I.shape[1])
    # train_L_R_flat = train_L_R
    # train_L_I_flat = train_L_I
    # train_L = np.concatenate((train_L_R_flat, train_L_I_flat), axis=1)
    # train_L = np.stack((train_L_R, train_L_I), axis=1)

    # train_H_R_flat = train_H_R.transpose((0, 2, 1)).reshape(-1, 1,train_H_R.shape[1])
    # train_H_I_flat = train_H_I.transpose((0, 2, 1)).reshape(-1, 1,train_H_I.shape[1])
    # train_H = np.concatenate((train_H_R_flat, train_H_I_flat), axis=1)
    #
    # # 测试集数据转换
    # val_L_R_flat = val_L_R.transpose((0, 2, 1)).reshape(-1, 1,val_L_R.shape[1])
    # val_L_I_flat = val_L_I.transpose((0, 2, 1)).reshape(-1, 1,val_L_I.shape[1])
    # val_L = np.concatenate((val_L_R_flat, val_L_I_flat), axis=1)
    #
    #
    # val_H_R_flat = val_H_R.transpose((0, 2, 1)).reshape(-1, 1,val_H_R.shape[1])
    # val_H_I_flat = val_H_I.transpose((0, 2, 1)).reshape(-1, 1,val_H_I.shape[1])
    # val_H = np.concatenate((val_H_R_flat, val_H_I_flat), axis=1)
    train_L = np.stack((train_L_R, train_L_I), axis=1)
    # train_H = np.stack((train_H_R, train_H_I), axis=1)
    train_H = np.concatenate((train_H_R, train_H_I), axis=1)
    val_L = np.stack((val_L_R, val_L_I), axis=1)
    # val_H = np.stack((val_H_R, val_H_I), axis=1)
    val_H = np.concatenate((val_H_R, val_H_I), axis=1)
    print("处理前的维度")
    print("训练集：",train_L_R.shape)  # 输出：(16000, 200)
    print("验证集：",val_L_R.shape)  # 输出：(16000, 200)
    print("标签：",train_H_R.shape)  # 输出：(16000, 200)
    print("处理后的维度")
    print("训练集：",train_L.shape)  # 输出：(16000, 200)
    print("验证集：",val_L.shape)  # 输出：(16000, 200)
    print("标签：",train_H.shape)  # 输出：(16000, 200)


    # 清除不需要的变量
    # del train_L_R, train_L_I, train_H_R, train_H_I, train_L_R_flat, train_L_I_flat,train_H_R_flat,train_H_I_flat
    # del val_L_R, val_L_I, val_H_R, val_H_I,val_L_R_flat,val_L_I_flat,val_H_R_flat,val_H_I_flat

    # 添加时间戳
    nowTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 现在

    # 获取归一化工具（对每一列进行归一化，也就是每一个特征维度）
    mm_scalerRT_INPUT_H = preprocessing.MinMaxScaler()
    mm_scalerRT_INPUT_L = preprocessing.MinMaxScaler()

    # 将形状 (B, 2, H) 转换为 (B, 2*H)
    H_L = train_L.shape[2]
    H_H = 20
    train_L_reshaped = train_L.reshape(train_L.shape[0], -1)
    train_H_reshaped = train_H.reshape(train_H.shape[0], -1)
    val_L_reshaped = val_L.reshape(val_L.shape[0], -1)
    val_H_reshaped = val_H.reshape(val_H.shape[0], -1)

    # Normalization，分别对两个尺寸的阵列进行标准化，用训练集进行fit，测试集只进行transform
    train_L = mm_scalerRT_INPUT_L.fit_transform(train_L_reshaped)

    val_L = mm_scalerRT_INPUT_L.transform(val_L_reshaped)

    # Normalization
    train_H = mm_scalerRT_INPUT_H.fit_transform(train_H_reshaped)

    val_H = mm_scalerRT_INPUT_H.transform(val_H_reshaped)

    train_L = train_L.reshape(train_L.shape[0], 2,H_L)
    # train_H = train_H.reshape(train_H.shape[0], 2, H_H)
    train_H = train_H.reshape(train_H.shape[0], -1)
    val_L = val_L.reshape(val_L.shape[0], 2, H_L)
    # val_H = val_H.reshape(val_H.shape[0], 2, H_H)
    val_H = val_H.reshape(val_H.shape[0], -1)
    # 创建每次训练的文件夹。保存归一化工具和训练模型
    # dirStore = f"./Train"+ nowTime
    # dirStore = f"./Train_CovmatrixCNN_"+ nowTime
    # 创建每次训练的文件夹。保存归一化工具和训练模型
    dirStore = f"./" + file_name + "_CNN_" + nowTime

    # 创建保存模型的文件夹
    if not os.path.exists(dirStore):
        os.makedirs(dirStore)

    # 保存归一化工具
    joblib.dump(mm_scalerRT_INPUT_L, os.path.join(dirStore, "Train_Array_scalarL_CNN"))
    joblib.dump(mm_scalerRT_INPUT_H, os.path.join(dirStore, "Train_Array_scalarH_CNN"))


    # 将数据reshape成一个二维数组，因为归一化对矩阵作用的话，是对每一列进行的
    # train_L = train_L.reshape(-1, 2, ArrayL, snap)
    # train_H = train_H.reshape(-1, 2, ArrayH, snap)
    # val_L = val_L.reshape(-1, 2, ArrayL, snap)
    # val_H = val_H.reshape(-1, 2, ArrayH, snap)

    # print("{:.16f}".format(train_L_R[0,0,0]))
    # 创建训练集和测试集的 Dataset
    train_dataset = MyDataset(train_L, train_H)
    val_dataset = MyDataset(val_L, val_H)
    # 将整个数据集加载到 GPU 内存中
    # train_dataset_on_gpu = [data.to(device) for data in train_dataset]
    # val_dataset_on_gpu = [data.to(device) for data in val_dataset]
    # L_R, L_I, H_R, H_I = train_dataset[0]
    # print(train_dataset[0])
    # print("{:.16f}".format(L_R[0,0]))
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # 实例化模型并将其移动到 GPU
    model = OriSigCNN(H_L,2*H_H,2).to(device)
    # 打印模型结构
    print(model)
    # 初始化模型权重
    model.apply(initialize_weights)
    # 初始化模型权重
    # model.apply(initialize_weights_xavier)

    # 创建一个随机输入张量
    # input_tensor = torch.randn(1, 2, ArrayL, snap).to(device)  # 假设输入图片大小为 32x32，通道数为 3
    input_tensor = torch.randn(1, train_L.shape[1],train_L.shape[2]).to(device)  # 假设输入图片大小为 32x32，通道数为 3
    # 通过模型前向传播计算输出
    output_tensor = model(input_tensor)

    # 打印输入和输出张量的大小
    print("Input tensor size:", input_tensor.size())
    print("Output tensor size:", output_tensor.size())


    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)  # 每10个epoch学习率减少为原来的0.1倍

    # 创建TensorBoard的SummaryWriter对象
    logdir = "runs2/"+ dirStore
    writer = SummaryWriter(log_dir=logdir)


    all_losses = []
    val_losses = []
    # # 训练模型
    # for epoch in range(num_epochs):
    #     train_loss = 0
    #     model.train()  # 设置模型为训练模式
    #     loop_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch [{epoch+1}/{num_epochs}]')
    #     for batch_idx,(data, target) in loop_train:
    #         data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         # 反向传播更新参数
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 累加训练损失
    #         train_loss += loss.item()
    #         # 在EXCEL中也保存一份
    #         # 保存损失值
    #         all_losses.append(loss.item())
    #         # 将训练损失写入TensorBoard
    #         writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    #         # 每10个批次更新一次进度条
    #         if batch_idx % 10 == 0:
    #             loop_train.set_postfix({'loss' : '{0:1.5f}'.format(loss.item())})
    #             # writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    #     # 学习率调度器在每个epoch结束后调用
    #     # scheduler.step()
    #     # 记录每个epoch的平均训练损失
    #     avg_train_loss = train_loss / len(train_loader)
    #     writer.add_scalar('AVG Training Loss', avg_train_loss, epoch)
    #
    #     # 测试模型
    #     model.eval()  # 设置模型为评估模式
    #     val_loss = 0
    #     with torch.no_grad():
    #         loop_val = tqdm(enumerate(val_loader), total=len(val_loader),desc=f'Val   Epoch [{epoch+1}/{num_epochs}]')
    #         for batch_idx,(data, target) in loop_val:
    #             data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
    #             output = model(data)
    #             # 计算验证集的loss
    #             temp_loss = criterion(output, target)
    #             val_loss += temp_loss.item()
    #             val_losses.append(temp_loss.item())
    #             # # 将训练损失写入TensorBoard
    #             writer.add_scalar('Val Loss', temp_loss.item(), epoch * len(val_loader) + batch_idx)
    #             # 每10个批次更新一次进度条
    #             if batch_idx % 10 == 0:
    #                 # writer.add_scalar('Val Loss', temp_loss.item(), epoch * len(val_loader) + batch_idx)
    #                 # loop_val.set_postfix(loss=temp_loss.item())
    #                 loop_val.set_postfix({'loss': '{0:1.5f}'.format(temp_loss.item())})
    #     # 记录每个epoch的平均验证损失
    #     avg_val_loss = val_loss / len(val_loader)
    #     writer.add_scalar('AVG Val Loss', avg_val_loss, epoch)
    #
    # writer.close()
    #
    # # 保存损失值到 Excel 文件
    # # 将数据转换为 Pandas Series
    # train_loss_series = pd.Series(all_losses, name='train_loss')
    # val_loss_series = pd.Series(val_losses, name='val_loss')
    #
    # # 使用 concat 函数将它们合并成一个 DataFrame，Pandas 会自动对齐索引
    # data = pd.concat([train_loss_series, val_loss_series], axis=1)
    #
    # # 保存到 CSV 文件
    # data.to_csv(dirStore + '/losses.csv', index=False)

    best_val_loss = float('inf')  # 初始化最好的验证损失为正无穷

    # 训练模型
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()  # 设置模型为训练模式
        loop_train = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'训练 Epoch [{epoch + 1}/{num_epochs}]')
        for batch_idx, (data, target) in loop_train:
            data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_losses.append(loss.item())
            writer.add_scalar('训练损失', loss.item(), epoch * len(train_loader) + batch_idx)

            if batch_idx % 10 == 0:
                loop_train.set_postfix({'损失': '{0:1.5f}'.format(loss.item())})

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('平均训练损失', avg_train_loss, epoch)

        # 测试模型
        model.eval()  # 设置模型为评估模式
        val_loss = 0
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'验证 Epoch [{epoch + 1}/{num_epochs}]')
            for batch_idx, (data, target) in loop_val:
                data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
                output = model(data)
                temp_loss = criterion(output, target)
                val_loss += temp_loss.item()
                val_losses.append(temp_loss.item())
                writer.add_scalar('验证损失', temp_loss.item(), epoch * len(val_loader) + batch_idx)

                if batch_idx % 10 == 0:
                    loop_val.set_postfix({'损失': '{0:1.5f}'.format(temp_loss.item())})

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('平均验证损失', avg_val_loss, epoch)

        # 如果当前验证损失小于最好的验证损失，则保存模型状态
        if avg_val_loss < best_val_loss:
            # best_val_loss = avg_val_loss
            # # 将模型转到CPU上并保存最佳模型的参数
            # model.cpu()
            # filename = dirStore + f"/{file_name}_CNN"
            # torch.save(model.state_dict(), filename + '.pth')
            # print(f"{filename} 保存成功！验证损失: {best_val_loss:.5f}")
            # # 再次将模型转回设备上
            # model.to(device)
            best_val_loss = avg_val_loss
            best_model_path = f"{dirStore}/{file_name}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"模型参数已保存到 {best_model_path} (最低验证损失: {best_val_loss:.5f})")

    writer.close()

    # 保存损失值到 CSV 文件
    train_loss_series = pd.Series(all_losses, name='train_loss')
    val_loss_series = pd.Series(val_losses, name='val_loss')
    data = pd.concat([train_loss_series, val_loss_series], axis=1)
    data.to_csv(dirStore + '/losses.csv', index=False)

    # # 将模型转到CPU上
    # model.cpu()
    # # 保存模型的参数
    # filename = dirStore + f"/"+file_name+"_CNN"
    # torch.save(model.state_dict(), filename + '.pth')
    # print(filename + "保存成功！凸(艹皿艹 )")
    # 将模型转到CPU上
    model.cpu()
    # 保存最终的模型参数
    final_model_path = f"{dirStore}/{file_name}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型参数已保存到 {final_model_path}")