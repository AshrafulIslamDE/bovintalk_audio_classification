import torch.nn as nn
import torch

from config import INPUT_SIZE, N_MELS

class RNN(nn.Module):
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

class LSTM(nn.Module):
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


class BiLSTM(nn.Module):
    def __init__(self, hidden_size=128, input_size=N_MELS, num_classes=2, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 1. Set bidirectional=True
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 2. The linear layer must take (hidden_size * 2) because
        # forward and backward states are concatenated.
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        # out shape: (batch, seq_len, hidden_size * 2)
        # hn shape: (num_layers * 2, batch, hidden_size)
        out, (hn, cn) = self.lstm(x)

        # 3. Extract the last hidden state for both directions
        # For a bidirectional LSTM, the last layer's hidden states are
        # at index -2 (forward) and -1 (backward).
        forward_last = hn[-2, :, :]
        backward_last = hn[-1, :, :]

        # Concatenate them to match the fc input features (hidden_size * 2)
        out = torch.cat((forward_last, backward_last), dim=1)

        out = self.fc(out)
        return out


class GRU(nn.Module):
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


