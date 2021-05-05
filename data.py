import re
from os import listdir
from os.path import join, getsize

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from zstandard import ZstdDecompressor

import config


class Collate(object):
    alphabet_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,
                       'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'v': 20,
                       'u': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

    num_to_alphabet = dict((v, k) for k, v in alphabet_to_num.items())

    def __init__(self, min_noise: int, max_noise: int):
        self.min_noise = max(0, min_noise)
        self.max_noise = max_noise % len(self.alphabet_to_num)

    def __call__(self, batch: list) -> torch.tensor:
        batch = [list(entry) for entry in batch]
        sizes = [len(entry) for entry in batch]
        batch_size, seq_len, token_size = len(batch), max(sizes), len(self.alphabet_to_num)

        src = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)
        tgt_inp = torch.zeros((batch_size, seq_len, token_size), dtype=torch.float)
        tgt = list()
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        for i in range(len(batch)):

            i_tgt = torch.full((seq_len,), fill_value=-1, dtype=torch.long)

            for j in range(len(batch[i])):
                num_repr = self.alphabet_to_num[batch[i][j]]

                src[i, j, num_repr] = 1
                tgt_inp[i, j, num_repr] = 1
                i_tgt[j] = num_repr

                noise_size = np.random.randint(low=self.min_noise, high=self.max_noise, size=1)[0]
                noise_indexes = np.random.randint(low=0, high=len(self.alphabet_to_num), size=noise_size)

                src[i, j, noise_indexes] = 1

            tgt.append(i_tgt)
            padding_mask[i, sizes[i]:] = True

        empty_token = torch.zeros(batch_size, 1, token_size)
        tgt_inp = torch.cat([empty_token, tgt_inp[:, :-1, :]], dim=1)
        tgt = torch.stack(tgt)
        subsequent_mask = self.get_subsequent_mask(seq_len)

        return src, tgt_inp, tgt, padding_mask, padding_mask, subsequent_mask

    @staticmethod
    def get_subsequent_mask(size: int) -> torch.tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.float), diagonal=1) == 1


class PileDataset(Dataset):

    def __init__(self, filenames: list, min_threshold: int = 150, max_threshold: int = 200,
                 drop_threshold: float = 0.62, dataset_size: int = 16_384):
        self.filenames = filenames
        self.n_files = len(self.filenames)
        self.file_sizes = [getsize(file) for file in self.filenames]
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.drop_threshold = drop_threshold
        self.dataset_size = dataset_size

    def __getitem__(self, idx=None):
        file_id = np.random.randint(low=0, high=self.n_files, size=1)[0]
        shift = np.random.randint(low=0, high=self.file_sizes[file_id] - self.max_threshold, size=1)[0]
        line_size = np.random.randint(low=self.min_threshold, high=self.max_threshold, size=1)[0]

        try:
            with open(self.filenames[file_id], mode="rb") as f:
                with ZstdDecompressor().stream_reader(f) as reader:
                    reader.seek(shift)
                    wiki_page = reader.read(line_size)

            sentence = wiki_page.decode("unicode_escape", errors='ignore')

            sentence = re.sub(r'\s', '', sentence)
            cleaned_sentence = re.sub(r'[^A-Za-z]', '', sentence)

            if len(sentence) == 0 or len(cleaned_sentence) / len(sentence) < self.drop_threshold:
                # print('drop_threshold', len(cleaned_sentence) / len(sentence), sentence)
                return self.__getitem__()

            # print(sentence)
            return cleaned_sentence.lower()

        except EOFError:
            return self.__getitem__()

    def __len__(self):
        return self.dataset_size


class WikiDataset(Dataset):

    def __init__(self, filenames: list, min_threshold: int = 150, max_threshold: int = 200, dataset_size: int = 16_384):
        self.filenames = filenames
        self.n_files = len(self.filenames)
        self.file_sizes = [getsize(file) for file in self.filenames]
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.dataset_size = dataset_size

    def __getitem__(self, idx=None):
        np.random.seed(None)

        file_id = np.random.randint(low=0, high=self.n_files, size=1)[0]
        shift = np.random.randint(low=0, high=self.file_sizes[file_id] - self.max_threshold, size=1)[0]
        line_size = np.random.randint(low=self.min_threshold, high=self.max_threshold, size=1)[0]

        with open(self.filenames[file_id], mode="r") as f:
            f.seek(shift)

            return f.read(line_size)

    def __len__(self):
        return self.dataset_size


def clean_wiki_text(filename: str, drop_threshold: float = 0.62) -> None:
    result_size = 0

    with open(filename, mode='r') as wiki_file, open(f'{filename}.clean', mode='w') as writer:

        while True:
            try:
                line = wiki_file.readline()
            except UnicodeDecodeError:
                continue

            if not line:
                break

            if not line.strip() or line.startswith(' = '):
                continue

            cleaned_line = re.sub(r'[^A-Za-z]', '', line)

            if len(cleaned_line) / len(line) < drop_threshold:
                continue

            writer.write(cleaned_line.lower())
            result_size += len(cleaned_line)

    print(f'Result size: {result_size}')


if __name__ == '__main__':
    train_files = [join(config.train_path, file) for file in listdir(config.train_path)]
    dataset = WikiDataset(train_files, min_threshold=199, max_threshold=200)

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=Collate(max_noise=8))

    for _src, _tgt_inp, _tgt, _src_pad_mask, _tgt_inp_pad_mask, _tgt_inp_attn_mask in loader:
        print(f'| src: {_src.size()} '
              f'| tgt_inp: {_tgt_inp.size()} '
              f'| tgt: {_tgt.size()} '
              f'| src_pad_mask: {_src_pad_mask.size()} '
              f'| tgt_inp_pad_mask: {_tgt_inp_pad_mask.size()} '
              f'| tgt_inp_attn_mask: {_tgt_inp_attn_mask.size()}')
