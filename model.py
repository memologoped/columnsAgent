import math

import torch
import torch.nn as nn
from model_utils import TEncoder, TDecoder


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=150):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model + 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        for i in range(x.size(0)):
            x[i] += self.pe[:x[i].size(0), 1:]
        return self.dropout(x)


class Net(nn.Module):
    def __init__(self, embedding_size, size_ff_net, num_encoder_layers, num_decoder_layers, head):
        super(Net, self).__init__()

        self.pos_enc = PositionalEncoding(embedding_size)
        self.encoder = TEncoder(embedding_size, size_ff_net, num_encoder_layers, head)
        self.decoder = TDecoder(embedding_size, size_ff_net, num_decoder_layers, head)


    def forward(self, train_data):
        train_data = self.pos_enc(train_data)

        train_data = self.encoder(train_data)
        train_data = self.decoder(train_data)
