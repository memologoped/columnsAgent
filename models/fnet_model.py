import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Collate


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        x = position * div_term

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, num_features: int, expansion_factor: int, dropout: float):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.dropout1(F.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x: torch.tensor) -> torch.tensor:
    return torch.fft.fft2(x, dim=(-1, -2)).real


class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int, dropout: float):
        super().__init__()
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out


class FNet(nn.TransformerEncoder):
    def __init__(self, d_model: int = 256, expansion_factor: int = 4, dropout: float = 0.5, num_layers: int = 6):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ZReader(nn.Module):
    def __init__(self, token_size: int, pe_max_len: int, num_layers: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super(ZReader, self).__init__()
        self.scale = math.sqrt(d_model)
        self.token_size = token_size

        self.mapping = nn.Linear(in_features=token_size, out_features=d_model)

        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=pe_max_len)

        self.f_net = FNet(d_model=d_model, dropout=dropout, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                        dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)

        self.inv_mapping = nn.Linear(in_features=d_model, out_features=token_size)

        self.init_weights()

    @staticmethod
    def get_parameters() -> tuple:
        token_size = len(Collate.alphabet_to_num)
        pe_max_len = 1000
        num_layers = 6
        d_model = 512
        n_heads = 8
        d_ff = 2048
        dropout = 0.1
        return token_size, pe_max_len, num_layers, d_model, n_heads, d_ff, dropout

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.tensor, src_pad_mask: torch.tensor,
                tgt_inp: torch.tensor, tgt_attn_mask: torch.tensor, tgt_pad_mask: torch.tensor) -> torch.tensor:

        src = src.transpose(0, 1)
        tgt_inp = tgt_inp.transpose(0, 1)

        src = F.relu(self.mapping(src)) * self.scale
        src = self.pe(src)

        tgt_inp = F.relu(self.mapping(tgt_inp)) * self.scale
        tgt_inp = self.pe(tgt_inp)

        encoded_src = self.f_net(src)

        decoded = self.decoder(tgt=tgt_inp, memory=encoded_src, tgt_mask=tgt_attn_mask,
                               tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)

        return self.inv_mapping(decoded).transpose(0, 1)

    def forward_read(self, src: torch.tensor, src_pad_mask: torch.tensor, tgt_inp: torch.tensor,
                     tgt_attn_mask: torch.tensor, tgt_pad_mask: torch.tensor) -> torch.tensor:
        src = src.transpose(0, 1)
        tgt_inp = tgt_inp.transpose(0, 1)

        src = F.relu(self.mapping(src)) * self.scale
        src = self.pe(src)

        encoded_src = self.f_net(src)

        for i in range(src.shape[1]):
            tgt = copy.deepcopy(tgt_inp)
            tgt = F.relu(self.mapping(tgt)) * self.scale
            tgt = self.pe(tgt)

            decoded = self.decoder(tgt=tgt_inp, memory=encoded_src, tgt_mask=tgt_attn_mask,
                                   tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)

            output = self.inv_mapping(decoded)
            output_tgt = copy.deepcopy(output[i:i + 1])

            predicted_characters = torch.zeros(output_tgt.shape)

            for i in range(0, output_tgt.shape[0]):
                for j in range(0, output_tgt.shape[1]):
                    pos = torch.argmax(output_tgt[i][j]).item()
                    predicted_characters[i][j][pos] = 1

            tgt_inp[i:i + 1] = predicted_characters

        return output.transopse(0, 1)

    def z_read(self, batch: list) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src, tgt_inp, src_pad_mask, tgt_inp_pad_mask, tgt_inp_attn_mask = self.batch_prepare(batch)
        with torch.no_grad():
            src = src.to(device)
            tgt_inp = tgt_inp.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_inp_pad_mask.to(device)
            tgt_attn_mask = tgt_inp_attn_mask.to(device)

            tgt_out = self.forward_read(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)

        for i in range(0, tgt_out.shape[0]):
            sent = ""
            for j in range(0, tgt_out.shape[1]):
                pos = torch.argmax(tgt_out[i][j]).item()
                sent += Collate.num_to_alphabet[pos]
            print(sent)

    @staticmethod
    def batch_prepare(batch: list) -> torch.tensor:
        sizes = [len(entry.split()) for entry in batch]
        batch_size, seq_len, token_size = len(batch), max(sizes), len(Collate.alphabet_to_num)
        src = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)
        tgt_inp = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)

        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        for i in range(len(batch)):
            for j, column in enumerate(batch[i].split(), start=0):
                for sym in column:
                    num_repr = Collate.alphabet_to_num[sym]
                    src[i, j, num_repr] = 1
            padding_mask[i, sizes[i]:] = True

        subsequent_mask = Collate.get_subsequent_mask(seq_len)

        return src, tgt_inp, padding_mask, padding_mask, subsequent_mask

    def save_parameters(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(filename, map_location=device))


if __name__ == "__main__":
    pass
