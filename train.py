from tqdm import tqdm
import glob
import os
import numpy as np
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from model.encoder import Encoder
from model.decoder import Decoder
from dataset import Dataset, _symbol_to_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
mel_scale = 4
dataset = Dataset('../DATASETS/LJSpeech-1.1/metadata.csv', '../DATASETS/LJSpeech-1.1/wavs',
                  mel_scale=mel_scale)
dataloader = DataLoader(dataset, collate_fn=dataset.collocate, batch_size=batch_size,
                        shuffle=True, num_workers=0)

writer = tensorboard.SummaryWriter(log_dir=f'logs')

for idx, batch in enumerate(dataloader):
    text_data, text_len, text_mask, mel_data, mel_len, mel_mask = batch
    # print(mel_data)
    mel_data = mel_data[0]
    mel_data = (mel_data.clamp(-15, 9) + 3) / 12.
    # mel_data = F.normalize(mel_data, 1)
    # mel_data = mel_data.clamp(-1, 1)
    # print(mel_data)
    writer.add_image(f'mel/target_', (mel_data.transpose(0, 1) + 1)/2., idx, dataformats='HW')
    writer.flush()
exit()
# -----------------------------------

embedding = nn.Embedding(num_embeddings=len(_symbol_to_id), embedding_dim=512)
encoder = Encoder(emb_channels=512, enc_out=512).to(device)
decoder = Decoder(mel_channels=80).to(device)

optimizer = torch.optim.Adam([{'params': embedding.parameters()},
                              {'params': encoder.parameters()},
                              {'params': decoder.parameters()}],
                             lr=0.001, betas=(0., 0.999))

# -----------------------------------

global_idx = 0
mean_losses = np.zeros(3)

for epoch in enumerate(tqdm(dataloader)):
    for idx, batch in enumerate(tqdm(dataloader)):
        text_data, text_len, text_mask, mel_data, mel_len, mel_mask = batch

        # audio_data = F.avg_pool1d(audio_data, kernel_size=2, padding=1)
        text_emb = embedding(text_data)
        x = encoder(text_emb, text_len)
        mels_out, gates_out, attentions_out = decoder(x, mel_data, text_mask, mel_mask)

        loss_mel = torch.mean((mels_out - mel_data) ** 2)
        loss_gate = F.binary_cross_entropy_with_logits(gates_out, mel_mask)
        loss = loss_mel + loss_gate

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -----------------------------------------
        global_idx += 1
        mean_losses += [loss_mel.item(),
                        loss_gate.item(),
                        loss.item()]
        if global_idx % 100:
            writer.add_images(f'mel/target', mel_data, global_idx)
            writer.add_images(f'mel/output', mels_out, global_idx)

            writer.add_images(f'mel/gate_target', mel_mask, global_idx)
            writer.add_images(f'mel/gate_out', gates_out, global_idx)

            writer.add_images(f'mel/attentions_out', mel_mask, global_idx)
