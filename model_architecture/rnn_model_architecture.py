import torch.nn as nn
import torch

from config import INPUT_SIZE


class RNN_LSTM(nn.Module):
    def __init__(self,hidden_size=128, input_size=INPUT_SIZE,num_classes=2,num_layers=2):
        super().__init__()
        self.num_layers =num_layers
        self.hidden_size = hidden_size
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=False)
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





class RNN_BiLSTM(nn.Module):
    def __init__(self, hidden_size=64, input_size=INPUT_SIZE, num_classes=2, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 1. Enable bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Enabled BiLSTM
        )

        # 2. Linear layer input is now hidden_size * 2
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 8, 13)
        batch_size, seq_len, window_size, mfcc = x.size()

        # Flatten windows and MFCC: (batch_size, seq_len, 104)
        x = x.view(batch_size, seq_len, -1)

        # 3. Initialize states: first dim must be num_layers * 2
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # out shape: (batch_size, seq_len, hidden_size * 2)
        out, _ = self.lstm(x, (h0, c0))

        # 4. Decode the hidden state of the LAST time step
        # For BiLSTM, 'out' already contains the concatenated forward/backward info
        out = self.fc(out[:, -1, :])

        return out


class RNN_GRU(nn.Module):
        def __init__(self, hidden_size=128, input_size=INPUT_SIZE, num_classes=2, num_layers=2):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )

            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, input):
            batch_size, seq_len, window_size, mfcc = input.size()

            # flatten the windows and mfcc
            input = input.view(batch_size, seq_len, -1)

            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device)

            # Forward propagate GRU
            # out: (batch_size, seq_len, hidden_size)
            out, _ = self.gru(input, h0)

            # Decode the hidden state of the LAST time step
            out = self.fc(out[:, -1, :])
            return out

class RNN_Vanilla(nn.Module):
    def __init__(self, hidden_size=128, input_size=INPUT_SIZE, num_classes=2, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'  # or 'relu'
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        batch_size, seq_len, window_size, mfcc = input.size()

        # flatten the windows and mfcc
        input = input.view(batch_size, seq_len, -1)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input.device)

        # Forward propagate RNN
        out, _ = self.rnn(input, h0)

        # Use the LAST time step
        out = self.fc(out[:, -1, :])
        return out