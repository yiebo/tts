from tqdm import tqdm
import glob
import os
import numpy as np
from prefetch_generator import BackgroundGenerator

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import Dataset, _symbol_to_id
import matplotlib.pyplot as plt
dataset = Dataset('F:/DATASETS/LJSpeech-1.1/metadata.csv', 'F:/DATASETS/LJSpeech-1.1/wavs',
                  mel_scale=1)
dataloader = DataLoader(dataset, collate_fn=dataset.collocate, batch_size=1,
                        shuffle=True, num_workers=0)

mel_list = np.zeros(1000000, dtype=float)
count = 0
with tqdm(total=1000000) as pbar:
    for idx, batch in enumerate(dataloader):
        print(idx)
        text_data, text_len, text_mask, mel_data, mel_len, mel_mask = batch
        # print(mel_data)
        for val in mel_data.flatten():
            if count == 1000000:
                plt.hist(mel_list, bins=1000)
                plt.show()
                exit()
            else:
                mel_list[count] = val
                count += 1
                pbar.update(1)

plt.hist(mel_list[:count], bins=1000)
plt.show()
exit()