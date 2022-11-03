import torch
import torch.nn as nn
import sys
import utils 

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32*48*7, 512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, seq_lengths):

        batch, max_seq_len, num_ch, in_dim = x.shape
        x = x.reshape(-1, num_ch, in_dim).unsqueeze(1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)

        out = out.reshape(batch*max_seq_len, -1)
        out = self.fc1(out)
        out = out.reshape(batch, max_seq_len, -1)

        lstm_out, _ = self.lstm(out)
        lstm_out = utils.last_relevant_pytorch(lstm_out, seq_lengths, batch_first=True)
        logits = self.fc2(lstm_out)

        return logits