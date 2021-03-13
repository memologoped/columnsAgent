"""the_pile dataset"""

import io
import os
import re

import jsonlines
import tensorflow as tf
import torch
import zstandard
from torch.utils.data import Dataset


def get_readers(filenames: list):
    readers = list()
    for file in filenames:
        with tf.io.gfile.GFile(file, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream)
            readers.append(reader)
    return readers


class PileData(Dataset):
    def __init__(self, readers: list):
        self.readers = readers
        self.index = 0
        self.result = None

    def __getitem__(self, index=None):

        assert self.index < 30, "No more file with data"

        try:
            line = self.readers[self.index].read()
            proc_line = self.text_prepare(line)
            return self.make_tensor(proc_line)
        except EOFError:
            self.index += 1
            self.__getitem__()

    def __len__(self):
        return 1000

    @classmethod
    def text_prepare(cls,  line):
        text_all = line["text"]
        text = text_all[:min(len(line['text']), 100)]
        result = str()
        sentences = text.split("\n\n")
        for sent in sentences:
            if len(sent) > 50:
                result += sent

        result = re.sub(r'[^A-Za-z]', '', result)
        result = result.lower()
        return result

    @classmethod
    def make_tensor(cls, line):
        result = list()
        alphabet = {'0': 0, 'a': 1/26, 'b': 2/26, 'c': 3/26, 'd': 4/26, 'e': 5/26, 'f': 6/26, 'g': 7/26, 'h': 8/26,
                    'i': 9/26, 'j': 10/26, 'k': 11/26, 'l': 12/26, 'm': 13/26, 'n': 14/26, 'o': 15/26, 'p': 16/26,
                    'q': 17/26, 'r': 18/26, 's': 19/26, 't': 20/26, 'v': 21/26, 'u': 22/26, 'w': 23/26, 'x': 24/26,
                    'y': 25/26, 'z': 1}
        for sym in line:
            result.append(alphabet[sym])
        return torch.tensor(result)

    def close_readers(self):
        for reader in self.readers:
            reader.close()


if __name__ == '__main__':
    train_dir = "H:\\data\\test"
    train_files = os.listdir(train_dir)
    print(train_files)
    train_files = [os.path.join(train_dir, str(el)) for el in train_files]

    pile_readers = get_readers(train_files)
    data = PileData(pile_readers)