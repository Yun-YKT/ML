import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_layer_size = 100,
                 output_size = 1, ngpus = 1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.ngpus = ngpus
        self.input_size = input_size


        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first = True)
        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq, hidden0=None):
        output, (hidden, cell) = self.lstm(input_seq, hidden0)
        output = self.linear(output[:, -1, :])

        return output
