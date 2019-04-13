import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalDropout(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, seq):
        ones = torch.ones_like(seq[:, :1, :])
        return seq * self.dropout(ones)


class HighwayNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.H = nn.Linear(input_size, input_size)
        self.T = nn.Linear(input_size, input_size)

    def forward(self, x):
        H = F.relu(self.H(x))
        T = torch.sigmoid(self.T(x))
        C = 1 - T

        return H * T + x * C


class Embedding(nn.Module):
    def __init__(self, word_vocab, char_vocab, char_conv_kernel_size, n_ctx_embs,
                 ctx_emb_dim):
        super().__init__()

        self.n_ctx_embs = n_ctx_embs

        word_dim, char_dim = word_vocab.emb_dim, char_vocab.emb_dim
        self.emb_dim = word_vocab.emb_dim + char_vocab.emb_dim + ctx_emb_dim

        if word_vocab.emb is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                torch.tensor(word_vocab.emb, dtype=torch.float32),
                freeze=word_vocab.freeze_emb)
        else:
            self.word_embedding = nn.Embedding(
                len(word_vocab), word_dim, padding_idx=word_vocab.sp.pad.idx)

        if char_vocab.emb is not None:
            self.char_embedding = nn.Embedding.from_pretrained(
                torch.tensor(char_vocab.emb, dtype=torch.float32),
                freeze=char_vocab.freeze_emb)
        else:
            self.char_embedding = nn.Embedding(
                len(char_vocab), char_dim, padding_idx=char_vocab.sp.pad.idx)

        if self.n_ctx_embs > 0:
            self.ctx_emb_weight = nn.Parameter(
                torch.full((n_ctx_embs, 1), 1 / n_ctx_embs))

        pad_size = ((char_conv_kernel_size - 1) // 2, char_conv_kernel_size // 2)
        self.const_pad1d = nn.ConstantPad1d(pad_size, 0)
        self.char_conv = nn.Conv1d(char_dim, char_dim, char_conv_kernel_size)
        self.highway = HighwayNetwork(self.emb_dim)

    def forward(self, x_word, x_char, ctx_emb):
        word_emb = self.word_embedding(x_word)

        char_emb = self.char_embedding(x_char)
        batch_size, seq_len, word_len, emb_dim = char_emb.shape
        char_emb = char_emb.transpose(2, 3).reshape(-1, emb_dim, word_len)
        char_emb = self.char_conv(self.const_pad1d(char_emb))
        char_emb = F.max_pool1d(char_emb, word_len, stride=1)
        char_emb = char_emb.squeeze().reshape(batch_size, seq_len, emb_dim)

        if self.n_ctx_embs > 0:
            ctx_emb = (ctx_emb * self.ctx_emb_weight).sum(dim=2)
            emb = torch.cat((word_emb, char_emb, ctx_emb), dim=-1)
        else:
            emb = torch.cat((word_emb, char_emb), dim=-1)

        emb = self.highway(emb)

        return emb


class Biattention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        score = x @ x.transpose(1, 2)
        score.masked_fill_(mask.unsqueeze(1) == 0, -np.inf)
        attention = F.softmax(score, dim=2)
        context = attention @ x

        return torch.cat((x, context, x + context, x - context, x * context), dim=-1)


class MixPooling(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, mask):
        x_max = torch.max(x, dim=1)[0]
        x_min = torch.min(x, dim=1)[0]
        x_avg = torch.mean(x, dim=1)

        score = self.linear(x)
        score.masked_fill_(mask.unsqueeze(2) == 0, -np.inf)
        attention = F.softmax(score, dim=1)
        x_attn = (attention * x).sum(dim=1)

        return torch.cat((x_max, x_min, x_avg, x_attn), dim=-1)


class MaxoutLinear(nn.Module):
    def __init__(self, input_size, output_size, n_units, dropout):
        super().__init__()

        self.output_size = output_size
        self.n_units = n_units
        self.linear = nn.Linear(input_size, n_units * output_size)
        self.dropout = nn.Dropout(dropout) if dropout != 0 else None

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, self.n_units, self.output_size)
        if self.dropout is not None:
            x = self.dropout(x)

        return torch.max(x, dim=1)[0]


class BatchNormMaxoutNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, n_units, dropout):
        super().__init__()

        layers = []
        for i, o in zip([input_size] + hidden_sizes, hidden_sizes + [output_size]):
            layers.append(nn.BatchNorm1d(i))
            layers.append(MaxoutLinear(i, o, n_units, dropout))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BCN(nn.Module):
    def __init__(self, word_vocab, char_vocab, char_conv_kernel_size, n_ctx_embs,
                 ctx_emb_dim, d_model, dropout):
        super().__init__()

        self.embedding = Embedding(
            word_vocab, char_vocab, char_conv_kernel_size, n_ctx_embs, ctx_emb_dim)
        self.dropout = VariationalDropout(dropout) if dropout != 0 else None
        self.encoder1 = nn.LSTM(
            self.embedding.emb_dim, d_model, batch_first=True, bidirectional=True)
        self.biattention = Biattention()
        self.encoder2 = nn.LSTM(
            d_model * 10, d_model, batch_first=True, bidirectional=True)
        self.mix_pooling = MixPooling(d_model * 2)
        self.maxout = BatchNormMaxoutNetwork(
            d_model * 8, [d_model * 4, d_model * 2], 5, 4, dropout)

    def forward(self, x_word, x_char, x_ctx_emb, pad_mask):
        x = self.embedding(x_word, x_char, x_ctx_emb)
        x, _ = self.encoder1(self.dropout(x) if self.dropout is not None else x)
        x = self.biattention(x, pad_mask)
        x, _ = self.encoder2(self.dropout(x) if self.dropout is not None else x)
        x = self.mix_pooling(x, pad_mask)
        x = self.maxout(x)

        return x
