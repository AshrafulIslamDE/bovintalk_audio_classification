import torch.nn as nn
import torch

class RNN_LSTM(nn.Module):
    def __init__(self,hidden_size=128, input_size=104,num_classes=2):
        super().__init__()
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc=nn.Linear(in_features=hidden_size,out_features=num_classes)

    def forward(self, input):
        batch_size, seq_len, window_size,mfcc=input.size()
        #flatten the windows and mfcc
        input=input.view(batch_size,seq_len,-1)
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(input, (h0, c0))

        # Decode the hidden state of the LAST time step only
        out = self.fc(out[:, -1, :])
        return out


