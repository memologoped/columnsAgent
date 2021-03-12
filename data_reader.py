"""the_pile dataset"""

import io
import os
import re

import jsonlines
import tensorflow as tf
import zstandard
from torch.utils.data import Dataset


class PileReader(object):
    def __init__(self, filenames: list):
        self.filenames = filenames

        self.readers = list()
        for file in self.filenames:
            self.readers.append(self.get_reader(file))

    def get_reader(self, filename):
        with tf.io.gfile.GFile(filename, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream)
        return reader


class PileData(Dataset):
    def __init__(self, readers: list):
        self.readers = readers
        self.index = 0
        self.result = None

    def __getitem__(self, index=None):

        if self.index > 29:
            raise Exception("no more data files")

        line = self.readers[self.index].read()
        proc_line = self.text_prepare(line)

        if line is not None:
            return proc_line
        else:
            self.index += 1
            line = self.readers[self.index].read()
            proc_line = self.text_prepare(line)
            return proc_line

    def __len__(self):
        return len(self.result)

    def text_prepare(self,  line):
        text = line["text"]
        result = str()
        sentences = text.split("\n\n")
        for sent in sentences:
            if len(sent) > 50:
                result += sent

        result = re.sub(r'[^A-Za-z]', '', result)
        result = result.lower()
        return result


if __name__ == '__main__':
    train_dir = "H:\\data\\train"
    train_files = os.listdir(train_dir)
    train_files = [train_dir + "\\" + str(el) for el in train_files]

    pile_readers = PileReader(train_files)
    data = PileData(pile_readers.readers)
