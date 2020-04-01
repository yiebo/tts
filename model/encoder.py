import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, emb_channels: int, enc_out=512):
        super().__init__()
        conv_layers = []
        for idx in range(3):
            block = nn.Sequential(nn.Conv1d(emb_channels, emb_channels, kernel_size=5, padding=5//2),
                                  nn.BatchNorm1d(emb_channels))
            conv_layers.append(block)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lstm = nn.LSTM(emb_channels, emb_channels//2, batch_first=True, bidirectional=True)

    def forward(self, text_embedding, text_len):
        """
        return:
            text_embedding: [B, T, C]
        """
        # [B, T, C] -> [B, C, T]
        x = text_embedding.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, text_len, batch_first=True)
        x = self.lstm(x)
        x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x
