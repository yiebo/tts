import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import Conv2dEqualized, LinearEqualized


class LocationBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=31):
        super().__init__()
        padding = kernel_size // 2
        self.conv1d = nn.Conv1d(2, 32, kernel_size=kernel_size, padding=padding)
        self.linear = nn.Linear(32, 128)

    def forward(self, x):
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, mel_channels, mel_scale):
        super().__init__()
        self.n_mel_dim = mel_channels * mel_scale

        self.prenet = nn.ModuleList([
            nn.Linear(self.n_mel_dim, 256),
            nn.Linear(256, 256),
        ])
        self.att_query_rnn = nn.LSTMCell(256 + 512, 1024)
        self.to_query = nn.Linear(1024, 128)

        self.memory_key = nn.Linear(512, 128)

        self.location_block = LocationBlock(128)

        self.attention = nn.Linear(128, 1)

        self.dec_rnn = nn.LSTMCell(512 + 1024, 1024)

        self.to_mel = nn.Linear(512 + 1024, self.n_mel_dim)
        self.to_gate = nn.Linear(self.n_mel_dim, 1)

    def forward(self, encoder_out, mels, mask):
        batch_size = encoder_out.size(0)
        time_size = encoder_out.size(1)
        mels_out, gates_out, attentions_out = [], [], []

        att_context = torch.zeros(batch_size, 512)
        att_hidden = torch.zeros(batch_size, 1024)
        att_cell = torch.zeros(batch_size, 1024)

        att_weight = torch.zeros(batch_size, 1, time_size)
        att_weight_total = torch.zeros(batch_size, 1, time_size)

        dec_hidden = torch.zeros(batch_size, 1024)
        dec_cell = torch.zeros(batch_size, 1024)

        # [B, T, C] -> [T, B, C]
        mel_init = torch.zeros(batch_size, 1, self.n_mel_dim)
        mels = torch.cat([mel_init, mels], 1).transpose(0, 1)
        mels = self.prenet(mels)

        # [B, T, 128]
        memory_keys = self.memory_key(encoder_out)

        # [T, B, C] -> T * [B, C]
        for mel in mels:
            # [B, 80 * self.n_frames_per_step + 512]
            x = torch.cat([mel, att_context], -1)
            att_hidden, att_cell = self.att_query_rnn(x, [att_hidden, att_cell])
            att_hidden = F.dropout(att_hidden, .5, self.train)
            # [B, 1024] -> [B, 1, 128]
            query = self.to_query(att_hidden.unsqueeze(1))

            # ------------------------------------------------------------------------------
            att_weight_ = torch.cat([att_weight, att_weight_total], 1)
            # [B, 2, T] -> [B, T, 128]
            location = self.location_block(att_weight_)

            # ------------------------------------------------------------------------------
            # [B, 1, 128] + [B, T, 128] + [B, T, 128] -> [B, T, 1] -> [B, 1, T]
            attention_total = F.tanh(query + memory_keys + location)
            attention_total = self.attention(attention_total).transpose(1, 2)

            # ------------------------------------------------------------------------------
            attention_total = torch.masked_fill(attention_total, mask, -float('Inf'))
            att_weight = F.softmax(attention_total, -1)

            # ------------------------------------------------------------------------------
            att_weight_total += att_weight
            # [B, 1, T] * [B, T, 512] -> [B, 512]
            att_context = torch.bmm(att_weight, encoder_out).squeeze(1)

            # ------------------------------------------------------------------------------
            att_context_hidden = torch.cat([att_context, att_hidden], -1)
            dec_hidden, dec_cell = self.dec_rnn(att_context_hidden, [dec_hidden, dec_cell])
            dec_hidden = F.dropout(dec_hidden, .5, self.train)

            dec_hidden = torch.cat([dec_hidden, att_context])
            mel_out = self.to_mel(dec_hidden)
            gate_out = self.to_gate(dec_hidden)

            mels_out.append(mel_out)
            gates_out.append(gate_out)
            attentions_out.append(att_weight)

        return mels_out, gates_out, attentions_out
