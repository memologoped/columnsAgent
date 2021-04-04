import os

import torch
from torch.utils.data import DataLoader
from data_reader import PileData, collate_fn_pad
from model import Net

# Define hyper-parameters
batch_size = 4
num_workers = 4
embedding_size = 27
sentence_max_len = 150
dropout = 0.1
num_encoder_layers = 6
num_decoder_layers = 6
head = 3
size_ff_net = 100


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train data prepare
train_dir = ".\\data"
train_files = os.listdir(train_dir)
train_files = [os.path.join(train_dir, str(el)) for el in train_files]

train_dataset = PileData(train_files)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True, drop_last=True, collate_fn=collate_fn_pad)

# Test data prepare
test_dir = ".\\data"
test_files = os.listdir(test_dir)
test_files = [os.path.join(test_dir, str(el)) for el in test_files]

test_dataset = PileData(test_files)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=collate_fn_pad)

model = Net(embedding_size, size_ff_net, num_encoder_layers, num_decoder_layers, head)


def train():
    for train_data, target in train_loader:
        output = model(train_data)


if __name__ == '__main__':
    train()