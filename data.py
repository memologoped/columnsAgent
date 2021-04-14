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

    def __init__(self, max_noise: int = 8):
        self.max_noise = max_noise % len(self.alphabet_to_num)

    def __call__(self, batch: list) -> torch.tensor:
        batch = [list(entry) for entry in batch]
        sizes = [len(entry) for entry in batch]
        max_len = max(sizes)
        tensor_shape = (len(batch), max_len, len(self.alphabet_to_num))

        numeric_batch = torch.zeros(tensor_shape, dtype=torch.float32)
        target = torch.zeros(tensor_shape, dtype=torch.float32)
        padding_mask = torch.zeros(tensor_shape[:-1], dtype=torch.bool)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                numeric_batch[i, j, self.alphabet_to_num[batch[i][j]]] = 1
                target[i, j, self.alphabet_to_num[batch[i][j]]] = 1

                noise_size = np.random.randint(low=0, high=self.max_noise, size=1)[0]
                noise_indexes = np.random.randint(low=0, high=len(self.alphabet_to_num), size=noise_size)

                numeric_batch[i, j, noise_indexes] = 1

            padding_mask[i, sizes[i]:] = True

        return numeric_batch, target, padding_mask


class WikiDataset(Dataset):

    def __init__(self, filenames: list, min_threshold: int = 150, max_threshold: int = 200,
                 drop_threshold: float = 0.62, dataset_size: int = 10_000):
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
                stream = ZstdDecompressor().stream_reader(f, read_size=int(shift + line_size))
                stream.seek(shift)

                wiki_page = stream.read(line_size)

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


def main():
    train_files = [join(config.data_path, file) for file in listdir(config.data_path)]
    dataset = WikiDataset(train_files)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True,
                        collate_fn=Collate(max_noise=8))

    for data, target, mask in loader:
        print(data.size())


if __name__ == '__main__':
    main()
