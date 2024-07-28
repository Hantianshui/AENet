# model = Sequential()
#
#         model.add(Dense(train_low.shape[1], input_dim=train_low.shape[1], activation='relu'))
#         model.add(Dense(int(train_low.shape[1]), activation='relu'))
#         model.add(Dense(int(train_high.shape[1]), activation='relu'))
#         model.add(Dense(train_high.shape[1], activation='relu'))
#         # compile the keras model
#         model.summary()
#         model.compile(optimizer='adam', loss='mean_squared_error')
#
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 200)               40200
# _________________________________________________________________
# dense_1 (Dense)              (None, 200)               40200
# _________________________________________________________________
# dense_2 (Dense)              (None, 512)               102912
# _________________________________________________________________
# dense_3 (Dense)              (None, 512)               262656
# =================================================================
# 根据参考的网络构建对应的pytorch网络实现对比
import torch.nn as nn
import torch
# class OriSigDNN(nn.Module):
#     def __init__(self,input_dim,output_dim):
#         super(OriSigDNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, 256)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.fc4 = nn.Linear(256, 256)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.fc5 = nn.Linear(256, output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = self.relu(self.bn1(self.fc1(x)))
#         # x = self.dropout(x)
#         x = self.relu(self.bn2(self.fc2(x)))
#         # x = self.dropout(x)
#         x = self.relu(self.bn3(self.fc3(x)))
#         # x = self.dropout(x)
#         x = self.relu(self.bn4(self.fc4(x)))
#         x = self.fc5(x)
#         return x


# class OriSigDNN(nn.Module):
#     def __init__(self,input_dim,output_dim, hidden_size=64, num_layers=2):
#         super(OriSigDNN, self).__init__()
#         # LSTM layers
#         self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
#
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_size, output_dim)
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_length, input_size)
#
#         # LSTM forward pass
#         lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size)
#
#         # Use the output of the last LSTM cell
#         last_time_step_output = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
#
#         # Fully connected layer
#         out = self.fc(last_time_step_output)  # shape: (batch_size, output_size)
#
#         return out
# self.fc1 = nn.Linear(input_dim * in_channel + self._to_linear * 3, 512)
# self.fc2 = nn.Linear(512, 256)
# self.fc3 = nn.Linear(256, 128)
# self.fc4 = nn.Linear(128, output_dim)
class OriSigDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OriSigDNN, self).__init__()
        self.flatten = nn.Linear(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, 5120)
        self.fc2 = nn.Linear(5120, 2560)
        self.fc3 = nn.Linear(2560, 1280)
        self.fc4 = nn.Linear(1280, 640)
        self.fc5 = nn.Linear(640, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

if __name__ == '__main__':
    net = OriSigDNN(input_dim = 20,output_dim = 40)
    input_tensor = torch.randn(1, 20)  # 示例输入，batch size = 1, 2 个通道，高度 10，宽度 100
    output = net(input_tensor)
    print(output.shape)  # 应输出 torch.Size([1, 2, 20, 100])