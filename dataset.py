import torch
from torch.nn.utils import rnn
from torch.utils import data
import torchaudio
from torchaudio import transforms

from text.cleaners import english_cleaners as clean_text
from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_suymbol = {i: s for i, s in enumerate(symbols)}


class Dataset(data.Dataset):
    def __init__(self, file_path, root_dir, mel_scale):
        with open(file_path, encoding='utf8') as file:
            self.data = [line.strip().split('|') for line in file]
        self.root_dir = root_dir
        self.resample = transforms.Resample(orig_freq=22050, new_freq=16000)
        self.to_mel = transforms.MelSpectrogram(n_mels=80, sample_rate=16000, n_fft=1024,
                                                hop_length=256, f_max=8000.)
        self.mel_scale = mel_scale
        self.text_pad = _symbol_to_id[' ']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        mel_data: [T, C]
        text_data: [T]
        """
        path, text = self.data[idx][0], self.data[idx][1]
        path = f'{self.root_dir}/{path}.wav'
        text = clean_text(text)
        text_sequence = []
        for s in text:
            if s in _symbol_to_id:
                text_sequence.append(_symbol_to_id[s])

        text_sequence = torch.tensor(text_sequence)
        data, sample_rate = torchaudio.load(path)
        audio_data = self.resample(data)

        mel_data = self.to_mel(audio_data)
        mel_data = torch.log(mel_data)
        mel_data = mel_data.transpose(1, 2).squeeze(0)

        # mel_data = (mel_data.clamp(-15, 9) + 3) / 12.

        return text_sequence, mel_data

    def collocate(self, batch):
        """
        batch: text_data: [T]], B * [mel_data: [T, C]
        -----
        return: text_data, text_len, text_mask, mel_data, mel_len
            text_data: [B, T], text_len: [B], mask: [B, T]
            mel_data: [B, T, C], mel_len: [B]
        """

        batch = sorted(batch, key=lambda x: x[1].size(0), reverse=True)

        text_data, mel_data = [], []
        for text, mel in batch:
            mel_data.append(mel)
            text_data.append(text)

        text_data = rnn.pack_sequence(text_data)
        text_data, text_len = rnn.pad_packed_sequence(text_data, batch_first=True, padding_value=self.text_pad)

        # pad so it is scalable
        mel_max_len = mel_data[0].size(0)
        mel_max_len += self.mel_scale - (mel_max_len % self.mel_scale)

        mel_data = rnn.pack_sequence(mel_data)
        mel_data, mel_len = rnn.pad_packed_sequence(mel_data, batch_first=True,
                                                    padding_value=0.0, total_length=mel_max_len)

        text_mask = torch.arange(text_len[0]).unsqueeze(0) > text_len.unsqueeze(-1)
        mel_mask = torch.arange(mel_len[0]).unsqueeze(0) > mel_len.unsqueeze(-1)

        return text_data, text_len, text_mask, mel_data, mel_len, mel_mask
