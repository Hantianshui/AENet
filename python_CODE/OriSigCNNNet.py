# # 初始卷积层，提取特征
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(ArrayL, ArrayL, 2)))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# # 最大池化层
# # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # 上采样层，将图像尺寸增加到20x20
# model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# # 再次上采样层，保持图像尺寸不变
# # model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# # 输出层，匹配目标尺寸20x20x2，使用Sigmoid是假设输出范围为[0,1]
# model.add(Conv2D(2, (3, 3), padding='same', activation='sigmoid'))
import torch.nn as nn
import torch
import torch.nn.functional as F
# class OriSigCNN(nn.Module):
#     def __init__(self):
#         super(OriSigCNN, self).__init__()
#         # 第一层卷积，提取特征
#
#         self.in_layer = nn.Linear(2 * 10, 125 * 8, bias=False)
#         self.conv1 = nn.Conv1d(8, 16, kernel_size = 3, padding = 1, bias=False,
#                   padding_mode='circular')
#         self.bn1 = nn.BatchNorm1d(16)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size = 3, padding = 1, bias=False,
#                   padding_mode='circular')
#         self.bn2 = nn.BatchNorm1d(32)
#         self.conv3 = nn.Conv1d(16, 8, kernel_size = 3, padding = 1, bias=False,
#                   padding_mode='circular')
#         self.bn3 = nn.BatchNorm1d(8)
#         # 第二层卷积，进一步提取特征
#         # self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 1), bias=False)
#         # 第三层卷积，减少通道数
#         # self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1), bias=False)
#         # 输出层，将通道数恢复到2
#         # self.conv4 = nn.Conv2d(32, 2, kernel_size=(3, 3), padding=(1, 1), bias=False)
#         # self.bn1 = nn.BatchNorm2d(128)
#         # 全连接层
#         self.fc1 = nn.Linear(8*125 , 128)
#         self.fc2 = nn.Linear(128, 64)
#         # self.fc3 = nn.Linear(256, 128)
#         # 输出层线性层
#         self.fc_out = nn.Linear(64, 2*20)  # 将输出尺寸调整为和期望输出一样
#
#         self.dropout = nn.Dropout(p=0.1)
#     def forward(self, x):
#         _x = x
#         bsz = x.size(0)
#         x = self.in_layer(x).view(bsz, 8, -1)
#         # print(x.size())
#         x = self.conv1(x)  # 第一层卷积
#         # print(x.size())
#         x = self.bn1(x)
#         x = F.relu(x)
#         # x = self.conv2(x)  # 第一层卷积
#         # # print(x.size())
#         # x = self.bn2(x)
#         # x = F.relu(x)
#         x = self.conv3(x)  # 第一层卷积
#         # print(x.size())
#         x = self.bn3(x)
#         x = F.relu(x)
#         # x = F.relu(self.conv2(x))  # 第二层卷积
#         # x = self.bn1(x)
#         # x = F.relu(self.conv3(x))  # 第三层卷积
#         # x = F.relu(self.conv4(x))  # 第三层卷积
#         # x = x + _x
#         x = x.view(bsz, -1)  # 展平特征图
#         x = F.relu(self.fc1(x))  # 全连接层
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))  # 全连接层
#         x = self.dropout(x)
#         # x = F.relu(self.fc3(x))  # 全连接层
#         # x = self.dropout(x)
#         x = self.fc_out(x)  # 输出层线性层
#         # 调整输出尺寸以匹配原始代码中的期望输出
#         x = x.view(bsz, 40)
#         return x
#
# class OriSigCNN(nn.Module):
#     def __init__(self):
#         super(OriSigCNN, self).__init__()
#         # 第一层卷积，提取特征
#
#         self.in_layer = nn.Linear(2 * 10, 125 * 8, bias=False)
#         self.conv1 = nn.Conv1d(8, 16, kernel_size = 3, padding = 1, bias=False,
#                   padding_mode='circular')
#         self.bn1 = nn.BatchNorm1d(16)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size = 3, padding = 1, bias=False,
#                   padding_mode='circular')
#         self.bn2 = nn.BatchNorm1d(32)
#         self.conv3 = nn.Conv1d(16, 8, kernel_size = 3, padding = 1, bias=False,
#                   padding_mode='circular')
#         self.bn3 = nn.BatchNorm1d(8)
#         # 第二层卷积，进一步提取特征
#         # self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 1), bias=False)
#         # 第三层卷积，减少通道数
#         # self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1), bias=False)
#         # 输出层，将通道数恢复到2
#         # self.conv4 = nn.Conv2d(32, 2, kernel_size=(3, 3), padding=(1, 1), bias=False)
#         # self.bn1 = nn.BatchNorm2d(128)
#         # 全连接层
#         self.fc1 = nn.Linear(8*125 , 128)
#         self.fc2 = nn.Linear(128, 64)
#         # self.fc3 = nn.Linear(256, 128)
#         # 输出层线性层
#         self.fc_out = nn.Linear(64, 2*20)  # 将输出尺寸调整为和期望输出一样
#
#         self.dropout = nn.Dropout(p=0.1)
#     def forward(self, x):
#         _x = x
#         bsz = x.size(0)
#         x = self.in_layer(x).view(bsz, 8, -1)
#         # print(x.size())
#         x = self.conv1(x)  # 第一层卷积
#         # print(x.size())
#         x = self.bn1(x)
#         x = F.relu(x)
#         # x = self.conv2(x)  # 第一层卷积
#         # # print(x.size())
#         # x = self.bn2(x)
#         # x = F.relu(x)
#         x = self.conv3(x)  # 第一层卷积
#         # print(x.size())
#         x = self.bn3(x)
#         x = F.relu(x)
#         # x = F.relu(self.conv2(x))  # 第二层卷积
#         # x = self.bn1(x)
#         # x = F.relu(self.conv3(x))  # 第三层卷积
#         # x = F.relu(self.conv4(x))  # 第三层卷积
#         # x = x + _x
#         x = x.view(bsz, -1)  # 展平特征图
#         x = F.relu(self.fc1(x))  # 全连接层
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))  # 全连接层
#         x = self.dropout(x)
#         # x = F.relu(self.fc3(x))  # 全连接层
#         # x = self.dropout(x)
#         x = self.fc_out(x)  # 输出层线性层
#         # 调整输出尺寸以匹配原始代码中的期望输出
#         x = x.view(bsz, 40)
#         return x

#   表格中的网络模型 # 单尺度模型
# class OriSigCNN(nn.Module):
#     def __init__(self, input_dim, output_dim, in_channel):
#         super(OriSigCNN, self).__init__()
#         # Define the convolutional layers
#         self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.relu = nn.ReLU()
#
#         # Calculate the size after conv and pooling layers
#         self._to_linear = 128 * (input_dim // 2 // 2 // 2)  # Account for three pooling layers
#
#         # Define the fully connected layers
#         self.flatten = nn.Linear(input_dim * in_channel, input_dim * in_channel)
#         self.fc1 = nn.Linear(input_dim * in_channel + self._to_linear, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#
#     def forward(self, x):
#         _x = x.view(x.shape[0], -1)  # Flatten the input
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)
#
#         x = x.view(-1, self._to_linear)  # Flatten the output from conv layers
#         # print(x.shape)
#         _x = self.flatten(_x)  # Flatten original input
#
#         x = torch.cat((_x, x), dim=1)  # Concatenate the flattened input and conv output
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

import torch
import torch.nn as nn
# 多尺度模型
# class OriSigCNN(nn.Module):
#     def __init__(self, input_dim, output_dim, in_channel):
#         super(OriSigCNN, self).__init__()
#         # Define the convolutional layers
#         self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU()
#
#         # Define pooling layers of different scales
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
#
#         # Calculate the size after conv and pooling layers
#         self._to_linear = 128 * (input_dim // 2)  # Account for three convolution layers and pooling
#
#         # Define the fully connected layers
#         self.flatten = nn.Linear(input_dim * in_channel, input_dim * in_channel)
#         self.fc1 = nn.Linear(input_dim * in_channel + self._to_linear * 3, 5120)
#         self.fc2 = nn.Linear(5120, 2560)
#         self.fc3 = nn.Linear(2560, 1280)
#         self.fc4 = nn.Linear(1280, output_dim)
#         # self.fc1 = nn.Linear(input_dim * in_channel + self._to_linear * 3, 512)
#         # self.fc2 = nn.Linear(512, 256)
#         # self.fc3 = nn.Linear(256, 128)
#         # self.fc4 = nn.Linear(128, output_dim)
#         self.drop = nn.Dropout(0.5)
#     def forward(self, x):
#         _x = x.view(x.shape[0], -1)  # Flatten the input
#
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#
#         # Pooling at different scales
#         x1 = self.pool1(x)
#         x2 = self.pool2(x)
#         x3 = self.pool3(x)
#         # print(x1.shape)
#         # print(x2.shape)
#         # print(x3.shape)
#         # Flatten the output from pooling layers
#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#         x3 = x3.view(x3.size(0), -1)
#         # print(x1.shape)
#         # print(x2.shape)
#         # print(x3.shape)
#         # Concatenate the flattened input and pooled outputs
#         _x = self.flatten(_x)  # Flatten original input
#         # print(_x.shape)
#         x = torch.cat((_x, x1, x2, x3), dim=1)
#
#         x = self.relu(self.fc1(x))
#         # x = self.drop(self.relu(self.fc2(x)))
#         # x = self.drop(self.relu(self.fc3(x)))
#         x = (self.relu(self.fc2(x)))
#         x = (self.relu(self.fc3(x)))
#         x = self.fc4(x)
#         return x



# 多卷积多尺度
class OriSigCNN(nn.Module):
    def __init__(self, input_dim, output_dim, in_channel):
        super(OriSigCNN, self).__init__()
        # Define the convolutional layers
        self.conv3 = nn.Conv1d(in_channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channel, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv11 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv22 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv33 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

        # Define pooling layers of different scales
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        # Calculate the size after conv and pooling layers
        # self._to_linear = 64 * (input_dim // 2 // 2) *3  # Account for three convolution layers and pooling
        self._to_linear = 64 * (2 + 3 + 2)  # Account for three convolution layers and pooling
        # Define the fully connected layers
        self.flatten = nn.Linear(input_dim * in_channel, input_dim * in_channel)
        self.fc1 = nn.Linear(input_dim * in_channel + self._to_linear, 5120)
        self.fc2 = nn.Linear(5120, 2560)
        self.fc3 = nn.Linear(2560, 1280)
        self.fc4 = nn.Linear(1280, output_dim)
        # self.fc1 = nn.Linear(input_dim * in_channel + self._to_linear * 3, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, output_dim)
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        x_ori = x
        _x = x.view(x.shape[0], -1)  # Flatten the input

        x1 = self.relu(self.conv1(x_ori))
        x2 = self.relu(self.conv2(x_ori))
        x3 = self.relu(self.conv3(x_ori))

        # Pooling at different scales
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x1 = self.relu(self.conv11(x1))
        x2 = self.relu(self.conv22(x2))
        x3 = self.relu(self.conv33(x3))
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # Flatten the output from pooling layers
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # Concatenate the flattened input and pooled outputs
        _x = self.flatten(_x)  # Flatten original input
        # print(_x.shape)
        x = torch.cat((_x, x1, x2, x3), dim=1)

        x = self.relu(self.fc1(x))
        # x = self.drop(self.relu(self.fc2(x)))
        # x = self.drop(self.relu(self.fc3(x)))
        x = (self.relu(self.fc2(x)))
        x = (self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


# # Example usage
# input_dim = 10  # Example input dimension (2x10 features, so the length is 10)
# output_dim = 10   # Example output dimension for classification
# in_channel = 2    # Number of input channels
#
# model = OriSigCNN(input_dim, output_dim, in_channel)
# print(model)



# class OriSigCNN(nn.Module):
#     def __init__(self, input_dim, output_dim, in_channel):
#         super(OriSigCNN, self).__init__()
#         # Define the convolutional layers
#         self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channel, 32, kernel_size=5, stride=1, padding=2, bias=False)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.relu = nn.ReLU()
#
#         # Additional convolutional layer with kernel size 1
#         self.conv1x1 = nn.Conv1d(in_channel, 32, kernel_size=1, stride=1, padding=0, bias=False)
#
#         # Calculate the size after conv and pooling layers
#         self._to_linear = 32 * (input_dim // 2)  # Account for three pooling layers
#
#         # Define the fully connected layers
#         # self.fc1 = nn.Linear(input_dim * in_channel + self._to_linear, 256)
#         self.fc1 = nn.Linear(3*self._to_linear, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#
#     def forward(self, x):
#         _x = self.relu(self.conv1x1(x))  # Apply conv1x1 to the input
#         _x = self.pool(_x)
#         _x = _x.view(_x.shape[0], -1)  # Flatten the result of conv1x1
#         # print(_x.shape)
#         x1 = self.relu(self.conv1(x))
#         x1 = self.pool(x1).view(_x.shape[0], -1)
#         # print(x1.shape)
#         x2 = self.relu(self.conv2(x))
#         x2 = self.pool(x2).view(_x.shape[0], -1)
#         # print(x2.shape)
#         # x = self.relu(self.conv3(x))
#         # x = self.pool(x)
#         # print(self._to_linear)
#         # x = x.view(-1, self._to_linear)  # Flatten the output from conv layers
#
#         x = torch.cat((_x, x1, x2), dim=1)  # Concatenate the flattened conv1x1 output and conv layers output
#         # print(x.shape)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# class OriSigCNN(nn.Module):
#     def __init__(self,input_dim,output_dim,in_channel,out_channel):
#         super(OriSigCNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         # 定义不同卷积层，使用不同的卷积核大小和步长
#         # self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=1, stride=1)
#         # self.conv2 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=2, stride=2)
#         # self.conv3 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=3, stride=3)
#         # self.conv4 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=4, stride=4)
#         # self.conv5 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=5, stride=5)
#         # 定义不同卷积层，使用不同的卷积核大小和步长
#         self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=1, stride=1, padding=0)
#         self.conv2 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=2, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=4, stride=1, padding=2)
#         self.conv5 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=5, stride=1, padding=2)
#         self.conv6 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=6, stride=1, padding=3)
#         self.conv7 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=7, stride=1, padding=3)
#         self.conv8 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=8, stride=1, padding=4)
#         self.conv9 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=9, stride=1, padding=4)
#         self.conv10 = nn.Conv1d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=10, stride=1, padding=5)
#         # self.conv6 = nn.Conv1d(in_channels=2 * in_channel, out_channels=4 * in_channel, kernel_size=3, stride=1, padding=1)
#         self.change = nn.Linear(420,1280)
#         self.change2 = nn.Conv1d(in_channels=2 * in_channel * 10, out_channels=256, kernel_size=1, stride=1, padding=0)
#         # 定义全连接层
#         # self.fc1 = nn.Linear(256*5, 2560)
#         # self.fc2 = nn.Linear(2560, 1280)
#         # self.fc3 = nn.Linear(1280, output_dim*out_channel)
#         self.fc1 = nn.Linear(256*5, input_dim*input_dim)
#         self.fc2 = nn.Linear(input_dim*input_dim, input_dim*input_dim)
#         # self.fc5 = nn.Linear(input_dim, 2*input_dim)
#         # self.fc6 = nn.Linear(2*input_dim, 2 * input_dim)
#         self.fc3 = nn.Linear(input_dim*input_dim, output_dim*output_dim)
#         self.fc4 = nn.Linear(output_dim*output_dim, output_dim*out_channel)
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # x 的形状为 (batch_size, channels, sequence_length)
#         B,C,N = x.size()
#         x1 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv1(x)))
#         x2 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv2(x)))
#         x3 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv3(x)))
#         x4 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv4(x)))
#         x5 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv5(x)))
#         x6 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv6(x)))
#         x7 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv7(x)))
#         x8 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv8(x)))
#         x9 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv9(x)))
#         x10 = nn.AdaptiveAvgPool1d(5)(F.relu(self.conv10(x)))
#         # print(x5.shape)
#         # 拼接卷积结果
#         x = torch.cat((x1, x2, x3, x4, x5,x6,x7,x8,x9,x10), dim=1)
#         # x = torch.cat((x1,  x3), dim=1)
#         # print(x.shape)
#         # 展平特征
#         # x_flat = x_concat.view(x_concat.size(0), -1)
#         # print(x_flat.shape)
#         x = (self.change2(x)).view(B,-1)
#         # print(x.shape)
#         # 应用全连接层
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.dropout(x)
#         # x = torch.relu(self.fc5(x))
#         # x = torch.relu(self.fc6(x))
#         x = torch.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc4(x))
#         output = x.view(B,C,self.output_dim)
#
#         return output

if __name__ == '__main__':
    net = OriSigCNN(10,40,2)
    input_tensor = torch.randn(1, 2, 10)  # 示例输入，batch size = 1, 2 个通道，高度 10，宽度 100
    output = net(input_tensor)
    print(output.shape)  # 应输出 torch.Size([1, 2, 20, 100])