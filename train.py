from os import listdir
from os.path import join

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import config
from data import WikiDataset, Collate
from model import ZReader
from datetime import datetime


def epoch_training(epoch_idx: int, z_reader: ZReader, train_loader: DataLoader, optimizer, device: torch.device,
                   log_interval: int, criterion: eval, printable: bool = True) -> None:
    z_reader.train()

    for batch_idx, (src, tgt_inp, tgt, src_pad_mask, tgt_pad_mask, tgt_attn_mask) in enumerate(train_loader, start=1):
        src = src.to(device)
        tgt_inp = tgt_inp.to(device)
        tgt = tgt.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        tgt_attn_mask = tgt_attn_mask.to(device)

        optimizer.zero_grad()

        tgt_out = z_reader(src, src_pad_mask, tgt_inp, tgt_attn_mask, tgt_pad_mask)
        tgt_out = tgt_out.view(-1, z_reader.token_size)

        loss = criterion(tgt_out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(z_reader.parameters(), max_norm=0.5)
        optimizer.step()

        if printable and (epoch_idx * batch_idx) % log_interval == 0:
            accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size(0)
            print(f'Train Batch: {epoch_idx * batch_idx:>7} | Loss: {loss.item():.6f} | Accuracy: {accuracy:.3f}')


def epoch_testing(epoch_idx: int, z_reader: ZReader, test_loader: DataLoader, device: torch.device, criterion: eval,
                  printable: bool = True) -> None:
    z_reader.eval()

    test_loss = 0
    test_accuracy = 0

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
            tgt_out = tgt_out.view(-1, z_reader.token_size)

            loss = criterion(tgt_out, tgt).item()
            accuracy = (torch.argmax(tgt_out, dim=1) == tgt).float().sum() / tgt.size[0]

            test_loss += loss
            test_accuracy += accuracy

            if printable:
                print(f'Test Batch: {epoch_idx * batch_idx:>7} | Loss: {loss:.6f} | Accuracy: {accuracy:.3f}')

        test_loss /= batch_idx
        test_accuracy /= batch_idx

        print(f'Test Average: {epoch_idx:>7} | Loss: {test_loss:.6f} | Accuracy: {test_accuracy:.3f}')


def epoch_visualization(epoch_idx: int, z_reader: ZReader, vis_loader: DataLoader, device: torch.device) -> None:
    z_reader.eval()

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

            # TODO print visual representation performance (convert tgt_out tensor into characters)


def save_parameters(epoch_idx: int, z_reader: ZReader) -> None:
    date = datetime.now()
    weights_name = f'{date.month}{date.day}{date.hour}{date.minute}_{epoch_idx}'

    z_reader.save_parameters(filename=join(config.weights_path, weights_name))


def train() -> None:
    torch.manual_seed(2531)

    # ---------------------------------------------DATA PARAMETERS------------------------------------------------------
    train_files = [join(config.train_path, file) for file in listdir(config.train_path)]
    test_files = [join(config.test_path, file) for file in listdir(config.test_path)]
    min_threshold = 150
    max_threshold = 200
    train_dataset_size = 4096
    test_dataset_size = 256
    vis_dataset_size = 8
    num_workers = 2
    max_noise = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------------------------------MODEL PARAMETERS------------------------------------------------------
    token_size = len(Collate.alphabet_to_num)
    pe_max_len = 1000
    num_layers = 4
    d_model = 512  # d_model % n_heads = 0
    n_heads = 8
    d_ff = 2048
    dropout = 0.2
    pre_trained = False
    # -----------------------------------------OPTIMIZATION PARAMETERS--------------------------------------------------
    criterion = CrossEntropyLoss(ignore_index=-1)
    lr = 5.0
    lr_step_size = 1
    gamma = 0.95
    # ------------------------------------------TRAIN LOOP PARAMETERS---------------------------------------------------
    n_epochs = 100_000
    batch_size = 8
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

    collate_fn = Collate(max_noise=max_noise)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=collate_fn)
    vis_loader = DataLoader(dataset=vis_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    z_reader = ZReader(token_size=token_size, pe_max_len=pe_max_len, num_layers=num_layers,
                       d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout).to(device)

    if pre_trained:
        z_reader.load_parameters('path/to/weights', device=device)

    optimizer = torch.optim.SGD(z_reader.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    try:
        for epoch_idx in range(1, n_epochs + 1):
            epoch_training(epoch_idx, z_reader, train_loader, optimizer, device, log_interval, criterion,
                           printable=train_printable)

            epoch_testing(epoch_idx, z_reader, test_loader, device, criterion, printable=test_printable)

            if epoch_idx % saving_interval == 0:
                epoch_visualization(epoch_idx, z_reader, vis_loader, device)

            if epoch_idx % saving_interval == 0:
                save_parameters(epoch_idx, z_reader)

            scheduler.step()

    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    train()
