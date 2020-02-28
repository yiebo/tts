from PIL import Image
import torch
from torch.utils import data
from text.cleaners import english_cleaners as clean_text
from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

class Dataset(data.Dataset):
    def __init__(self, file_path, root_dir, transform):
        with open(file_path, encoding='utf8') as file:
            self.data = [line.strip().split('|') for line in file]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, text = self.data[idx][0], self.data[idx][1]
        text = clean_text(text)
        sequence = []
        for s in text:
            if s in _symbol_to_id:
                sequence.append(_symbol_to_id[s])
        sequence = torch.int(sequence)

        return sequence
