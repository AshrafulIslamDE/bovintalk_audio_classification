import torch.nn as nn
import torch

from config import INPUT_SIZE, N_MELS


class RNN_LSTM_Spectogram(nn.Module):
    def __init__(self,hidden_size=128, input_size=N_MELS,num_classes=2,num_layers=2):
        super().__init__()
        self.num_layers =num_layers
        self.hidden_size = hidden_size
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=False)
        self.fc=nn.Linear(in_features=hidden_size,out_features=num_classes)


    def forward(self, input):
        out, (hn, cn) = self.lstm(input)

        # Take last hidden state
        out = hn[-1]
        out = self.fc(out)

        return out





class BiLSTM_Spectogram(nn.Module):
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
        out, (hn, cn) = self.lstm(x)

        # Take last hidden state
        out = hn[-1]
        out = self.fc(out)

        return out


class RNN_GRU_Spectogram(nn.Module):
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
            out, hn = self.gru(input)  # ✅ GRU returns only hn
            # hn shape: (num_layers, batch, hidden_size)
            out = hn[-1]  # last layer hidden state
            out = self.fc(out)

            return out


class RNN_Spectogram(nn.Module):
    def __init__(self, hidden_size=128, input_size=N_MELS, num_classes=2, num_layers=2):
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
        # x shape: (batch, time, features)

        out, hn = self.rnn(input)

        # hn shape: (num_layers, batch, hidden_size)
        last_hidden = hn[-1]  # Take last layer hidden state

        out = self.fc(last_hidden)
        return out