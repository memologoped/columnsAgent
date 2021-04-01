import os
import re

import numpy as np
import tensorflow as tf
import torch
import zstandard
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def collate_fn_pad(batch):
    sentences = [list(el) for el in batch]

    alphabet_to_num = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11,
                       'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'v': 21,
                       'u': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}

    num_representation = [torch.tensor([alphabet_to_num[symb] for symb in sent]) for sent in sentences]
    res = pad_sequence(num_representation).T

    target = res.tolist()
    train_data = res.tolist()

    for i in range(len(target)):
        for j in range(len(target[i])):
            target_column = [0 for _ in range(27)]
            target_column[target[i][j]] = 1
            target[i][j] = target_column

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            train_column = np.random.choice(2, 27, replace=True).tolist()
            train_column[train_data[i][j]] = 1
            train_data[i][j] = train_column

    return torch.tensor(train_data), torch.tensor(target)


class PileData(Dataset):

    def __init__(self, filenames: list, threshold: int = 50):
        self.filenames = filenames
        self.threshold = threshold
        self.checkpoint = [0] * len(filenames)

    def __getitem__(self, item=None):
        file_num = np.random.choice(len(self.filenames), 1)[0]
        len_line = np.random.choice([i for i in range(50, 150)], 1)[0]
        file_size = os.path.getsize(self.filenames[file_num])
        shift_size = np.random.choice(file_size - 200, 1)[0]

        try:
            with tf.io.gfile.GFile(self.filenames[file_num], "rb") as f:

                cctx = zstandard.ZstdDecompressor()

                stream = cctx.stream_reader(f, read_size=file_size)
                stream.seek(shift_size)

                wiki_page = stream.read(len_line)
                sentence = self.__text_prepare(wiki_page.decode("utf-8"))

                if len(sentence) < self.threshold:
                    return self.__getitem__()

                return sentence

        except EOFError:
            return self.__getitem__()

    def __len__(self):
        return 1000

    def __text_prepare(self, text: str):
        sentence = str()
        for line in text.split("\n\n"):
            if len(line) > self.threshold:
                sentence += line
        cleaned_sentence = re.sub(r'[^A-Za-z]', '', sentence)
        return cleaned_sentence.lower()


def main():
    train_dir = ".\\data"
    train_files = os.listdir(train_dir)
    train_files = [os.path.join(train_dir, str(el)) for el in train_files]
    data = PileData(train_files)

    loader = DataLoader(dataset=data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True,
                        collate_fn=collate_fn_pad)

    for data, target in loader:
        print(data.size())
        print(target.size())


if __name__ == '__main__':
    main()
