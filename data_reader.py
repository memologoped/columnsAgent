"""the_pile dataset"""

import io
import os
import re

import jsonlines
import tensorflow as tf
import torch
import zstandard
from torch.utils.data import Dataset


class PileData(Dataset):
    def __init__(self, filenames: list, threshold: int = 50):
        self.readers = self.__get_readers(filenames)
        self.threshold = threshold
        self.pile_index = 0
        self.alphabet_to_float = {'0': 0, 'a': 1 / 26, 'b': 2 / 26, 'c': 3 / 26, 'd': 4 / 26, 'e': 5 / 26, 'f': 6 / 26,
                                  'g': 7 / 26, 'h': 8 / 26, 'i': 9 / 26, 'j': 10 / 26, 'k': 11 / 26, 'l': 12 / 26,
                                  'm': 13 / 26, 'n': 14 / 26, 'o': 15 / 26, 'p': 16 / 26, 'q': 17 / 26, 'r': 18 / 26,
                                  's': 19 / 26, 't': 20 / 26, 'v': 21 / 26, 'u': 22 / 26, 'w': 23 / 26, 'x': 24 / 26,
                                  'y': 25 / 26, 'z': 1}

    def __getitem__(self, index: int = None) -> torch.tensor:

        assert self.pile_index < 30, "No more data files"

        try:
            wiki_page = self.readers[self.pile_index].read()
            sentence = self.__text_prepare(wiki_page["text"])

            return self.__make_tensor(sentence)

        except EOFError:
            self.pile_index += 1
            self.__getitem__()

    def __len__(self) -> int:
        return 1000

    @staticmethod
    def __get_readers(filenames: list) -> list:
        readers = list()

        for file in filenames:
            with tf.io.gfile.GFile(file, 'rb+') as f:
                cctx = zstandard.ZstdDecompressor()
                reader_stream = io.BufferedReader(cctx.stream_reader(f))
                reader = jsonlines.Reader(reader_stream)
                readers.append(reader)

        return readers

    def __text_prepare(self, text: dict) -> str:
        text = text[:100]  # TODO !!!

        sentence = str()
        for line in text.split("\n\n"):
            if len(line) > self.threshold:
                sentence += line

        cleaned_sentence = re.sub(r'[^A-Za-z]', '', sentence)

        return cleaned_sentence.lower()

    def __make_tensor(self, sentence: str) -> torch.tensor:
        float_representation = [self.alphabet_to_float[c] for c in sentence]

        return torch.tensor(float_representation)

    def close_readers(self) -> None:
        for reader in self.readers:
            reader.close()


if __name__ == '__main__':
    train_dir = "H:\\data\\test"
    train_files = os.listdir(train_dir)

    print(train_files)

    train_files = [os.path.join(train_dir, str(el)) for el in train_files]

    data = PileData(train_files)
