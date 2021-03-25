import os
import re

import numpy as np
import tensorflow as tf
import zstandard
from torch.utils.data import Dataset, DataLoader


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
    train_dir = "C:\\columnsAgent\\data"
    train_files = os.listdir(train_dir)
    train_files = [os.path.join(train_dir, str(el)) for el in train_files]
    data = PileData(train_files)

    loader = DataLoader(dataset=data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    for batch in loader:
        print(batch)


if __name__ == '__main__':
    main()
