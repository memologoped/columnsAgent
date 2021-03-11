"""the_pile dataset"""

import io
import os
import re

import jsonlines
import tensorflow as tf
import zstandard


class PileReader:
    def __init__(self, filenames, para_joiner='\n\n'):
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.para_joiner = para_joiner

    def _read_fn(self, filename):
        with tf.io.gfile.GFile(filename, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream)
            for item in reader:
                text = item["text"]
                result = str()
                sentences = text.split("\n\n")
                for sent in sentences:
                    if len(sent) > 50:
                        result += sent

                result = re.sub(r'[^A-Za-z]', '', result)
                result = result.lower()
                yield result

    def __iter__(self):
        for filename in self.filenames:
            return self._read_fn(filename)


if __name__ == '__main__':

    train_dir = "H:\\data\\train"
    train_files = os.listdir(train_dir)
    train_files = [train_dir + "\\" + str(el) for el in train_files]

    data = PileReader(train_files)
    for el in data:
        print(el)
