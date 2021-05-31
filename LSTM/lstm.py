import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, model_size = 1, hidden_layer_size = 100,
                 output_size = 1, batch_size = 32, ngpus = 1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(model_size, hidden_layer_size)
        self.ngpus = ngpus

        self.model_size = model_size 
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, self.batch_size, self.hidden_layer_size),
                            torch.zeros(1, self.batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(seq_len, batch_size, 1),
                                               self.hidden_cell) #lstmのデフォルトの入力サイズは(シーケンスサイズ、バッチサイズ、特徴量次元数)
        predictions = self.linear(self.hidden_cell[0].view(batch_size, -1))
        return predictions[:, 0]