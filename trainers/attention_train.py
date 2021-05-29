import math
from datetime import datetime
from os import listdir
from os.path import join
from time import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import config
from data import WikiDataset, Collate
from models.attention_model import ZReader


def epoch_training(offset: int, z_reader: ZReader, train_loader: DataLoader, optimizer, scheduler, device: torch.device,
                   log_interval: int, criterion: eval, printable: bool = True) -> None:
    z_reader.train()

    checkpoint = open(config.checkpoint_path, "a")

    if printable:
        str_ = f'\nLearning Rate: {round(scheduler.get_last_lr()[0], 5)}\n{"*" * 79}'
        checkpoint.write(str_ + '\n')
        print(str_)

    start_time = time()

    for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(train_loader, start=1):
        src = src.to(device)
        tgt_inp = tgt_inp.to(device)
        tgt = tgt.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        tgt_attn_mask = tgt_attn_mask.to(device)

        tgt_out = z_reader(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
        tgt_out = tgt_out.reshape(-1, z_reader.token_size)
        tgt = tgt.view(-1)

        loss = criterion(tgt_out, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if printable and (offset + batch_idx) % log_interval == 0:
            accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)

            str_ = f'Train Batch:  {offset + batch_idx:^7} | Loss: {loss.item():>10.6f} | Accuracy: {accuracy:>6.3f} | ' \
                   f'Elapsed: {time() - start_time:>7.3f}s'
            print(str_)

            start_time = time()
            checkpoint.write(str_ + '\n')

    checkpoint.close()


def epoch_testing(offset: int, z_reader: ZReader, test_loader: DataLoader, device: torch.device, criterion: eval,
                  printable: bool = True) -> None:
    z_reader.eval()

    if printable:
        print('-' * 79)

    test_loss = 0
    test_accuracy = 0
    start_time = time()

    checkpoint = open(config.checkpoint_path, "a")
    checkpoint.write("=" * 80 + '\n')

    with torch.no_grad():
        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(test_loader,
                                                                                                   start=1):
            src = src.to(device)
            tgt_inp = tgt_inp.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)
            tgt_attn_mask = tgt_attn_mask.to(device)

            tgt_out = z_reader(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
            tgt_out = tgt_out.reshape(-1, z_reader.token_size)
            tgt = tgt.view(-1)

            loss = criterion(tgt_out, tgt).item()
            accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)

            test_loss += loss
            test_accuracy += accuracy

            if printable:
                str_ = f'Test Batch:   {offset + batch_idx:^7} | Loss: {loss:>10.6f} | Accuracy: {accuracy:>6.3f} | ' \
                       f'Elapsed: {time() - start_time:>7.3f}s'
                print(str_)

                start_time = time()
                checkpoint.write(str_ + '\n')

        test_loss /= batch_idx
        test_accuracy /= batch_idx

        str_ = f'Test Average: {"":^7} | Loss: {test_loss:>10.6f} | Accuracy: {test_accuracy:>6.3f} |'

        print(str_)
        checkpoint.write(str_ + '\n')
        checkpoint.close()


def epoch_visualization(offset: int, z_reader: ZReader, vis_loader: DataLoader, device: torch.device) -> None:
    z_reader.eval()

    checkpoint = open(config.checkpoint_path, "a")
    checkpoint.write("=" * 80 + '\n')

    with torch.no_grad():
        for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(vis_loader,
                                                                                                   start=1):
            src = src.to(device)
            tgt_inp = tgt_inp.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)
            tgt_attn_mask = tgt_attn_mask.to(device)

            tgt_out = z_reader(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)

            for i in range(0, tgt_out.shape[0]):
                sent = ""
                for j in range(0, tgt_out.shape[1]):
                    pos = torch.argmax(tgt_out[i][j]).item()
                    sent += Collate.num_to_alphabet[pos]
                print(sent)
                checkpoint.write(sent + '\n')

        checkpoint.close()


def save_parameters(epoch_idx: int, z_reader: ZReader) -> None:
    checkpoint = open(config.checkpoint_path, "a")
    date = datetime.now()
    weights_name = f'{date.month:0>2}{date.day:0>2}_{date.hour:0>2}{date.minute:0>2}_{epoch_idx}'
    checkpoint.write(weights_name + '\n')
    checkpoint.close()

    z_reader.save_parameters(filename=join(config.weights_path, weights_name))


def train() -> None:
    torch.manual_seed(2531)

    # ---------------------------------------------DATA PARAMETERS------------------------------------------------------
    train_files = [join(config.train_path, file) for file in listdir(config.train_path)]
    test_files = [join(config.test_path, file) for file in listdir(config.test_path)]
    min_threshold = 200
    max_threshold = 201
    train_dataset_size = 8000
    test_dataset_size = 64
    vis_dataset_size = 32
    num_workers = 5
    min_noise = 1
    max_noise = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------------------------------MODEL PARAMETERS------------------------------------------------------
    token_size = len(Collate.alphabet_to_num)
    pe_max_len = 1000
    num_layers = 6
    d_model = 512  # d_model % n_heads = 0
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    pre_trained = True
    # -----------------------------------------OPTIMIZATION PARAMETERS--------------------------------------------------
    criterion = CrossEntropyLoss(ignore_index=-1)
    lr = 0.00001
    lr_step_size = 1
    gamma = 0.95
    # ------------------------------------------TRAIN LOOP PARAMETERS---------------------------------------------------
    n_epochs = 100_000
    batch_size = 32
    saving_interval = 1
    log_interval = 1
    vis_interval = 1
    train_printable = True
    test_printable = True
    # ------------------------------------------------------------------------------------------------------------------

    train_dataset = WikiDataset(filenames=train_files, min_threshold=min_threshold, max_threshold=max_threshold,
                                dataset_size=train_dataset_size)
    test_dataset = WikiDataset(filenames=test_files, min_threshold=min_threshold, max_threshold=max_threshold,
                               dataset_size=test_dataset_size)
    vis_dataset = WikiDataset(filenames=test_files, min_threshold=min_threshold, max_threshold=max_threshold,
                              dataset_size=vis_dataset_size)

    collate_fn = Collate(min_noise=min_noise, max_noise=max_noise)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=collate_fn)
    vis_loader = DataLoader(dataset=vis_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    z_reader = ZReader(token_size=token_size, pe_max_len=pe_max_len, num_layers=num_layers,
                       d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout).to(device)

    if pre_trained:
        z_reader.load_parameters(join(config.weights_path, '0529_1251_91'), device=device)

    optimizer = torch.optim.AdamW(z_reader.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    try:
        for epoch_idx in range(1, n_epochs + 1):
            offset = math.ceil((epoch_idx - 1) * train_dataset_size / batch_size)

            epoch_training(offset, z_reader, train_loader, optimizer, scheduler, device, log_interval, criterion,
                           printable=train_printable)

            epoch_testing(offset, z_reader, test_loader, device, criterion, printable=test_printable)

            if epoch_idx % vis_interval == 0:
                epoch_visualization(offset, z_reader, vis_loader, device)

            if epoch_idx % saving_interval == 0:
                save_parameters(epoch_idx, z_reader)

            # scheduler.step()

    except KeyboardInterrupt:
        print('Interrupted')

        if input('Save model? ') == 'y':
            save_parameters(-1, z_reader)


if __name__ == '__main__':
    train()
