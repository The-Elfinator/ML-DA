import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import Dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerRussianToEnglish(nn.Module):
    def __init__(self, vocab_size, model_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, pad_idx=0):
        super(TransformerRussianToEnglish, self).__init__()

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model=model_dim, dropout=dropout)
        self.transformer = Transformer(d_model=model_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.pos_encoding(self.embedding(src))
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        output = self.fc_out(output)
        return output


class TranslationDataset(Dataset):
    def __init__(self, tokenizer, pairs, max_length=128):

        self.tokenizer = tokenizer
        self.pairs = pairs
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        return src_text, tgt_text


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
