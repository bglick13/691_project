from models.double_decoder import MovieSequenceEncoder, MovieSequenceDataset
from data_util import process_movielens_ratings
import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def train_baseline():
    model: MovieSequenceEncoder = MovieSequenceEncoder(sequence_length=SEQUENCE_LENGTH,
                                                       dataset=dataset,
                                                       embedding_dim=embedding_dim, n_head=n_head, ff_dim=ff_dim,
                                                       n_encoder_layers=n_encoder_layers,
                                                       use_ratings=False,
                                                       use_users=False,
                                                       model_name=f'baseline_{SEQUENCE_LENGTH}')
    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 1) Train without ratings as a baseline
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    model.cuda()
    model.pretrain(dl, lr=lr, print_iter=100, epochs=epochs, mask_pct=mask_pct)
    torch.save(model, f'baseline.torch')


def train_with_ratings():
    model: MovieSequenceEncoder = MovieSequenceEncoder(sequence_length=SEQUENCE_LENGTH,
                                                       dataset=dataset,
                                                       embedding_dim=embedding_dim, n_head=n_head, ff_dim=ff_dim,
                                                       n_encoder_layers=n_encoder_layers,
                                                       use_ratings=True,
                                                       use_users=False,
                                                       model_name=f'ratings_{SEQUENCE_LENGTH}')
    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 1) Train without ratings as a baseline
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    model.cuda()
    model.pretrain(dl, lr=lr, print_iter=100, epochs=epochs, mask_pct=mask_pct)
    torch.save(model, f'ratings.torch')


def train_with_users():
    model: MovieSequenceEncoder = MovieSequenceEncoder(sequence_length=SEQUENCE_LENGTH,
                                                       dataset=dataset,
                                                       embedding_dim=embedding_dim, n_head=n_head, ff_dim=ff_dim,
                                                       n_encoder_layers=n_encoder_layers,
                                                       use_ratings=False,
                                                       use_users=True,
                                                       model_name=f'users_{SEQUENCE_LENGTH}')
    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 1) Train without ratings as a baseline
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    model.cuda()
    model.pretrain(dl, lr=lr, print_iter=100, epochs=epochs, mask_pct=mask_pct)
    torch.save(model, f'users.torch')


def train_with_both(f='both'):
    model: MovieSequenceEncoder = MovieSequenceEncoder(sequence_length=SEQUENCE_LENGTH,
                                                       dataset=dataset,
                                                       embedding_dim=embedding_dim, n_head=n_head, ff_dim=ff_dim,
                                                       n_encoder_layers=n_encoder_layers,
                                                       use_ratings=True,
                                                       use_users=True,
                                                       model_name=f'both_{SEQUENCE_LENGTH}')
    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 1) Train without ratings as a baseline
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    model.cuda()
    model.pretrain(dl, lr=lr, print_iter=100, epochs=epochs, mask_pct=mask_pct)
    torch.save(model, f'{f}.torch')


if __name__ == '__main__':
    PCT_OF_DATA = 0.15
    SEQUENCE_LENGTH = 50
    ratings = pd.read_csv('../data/ml-20m/ratings.csv')
    np.random.seed(42)
    idxs = np.random.choice(range(len(ratings)), int(len(ratings) * PCT_OF_DATA), replace=False)
    ratings = ratings.iloc[idxs, :]
    embedding_dim = 256
    n_head = 8
    ff_dim = 1024
    n_encoder_layers = 4
    batch_size = 512
    lr = 1.0e-4
    epochs = 20
    mask_pct = 0.1

    try:
        dataset = torch.load('dataset.torch')
    except:
        dataset = MovieSequenceDataset(ratings, sequence_length=SEQUENCE_LENGTH, test_pct=0.1)
        torch.save(dataset, 'dataset.torch')
    dataset.train()

    # train_baseline()
    # train_with_ratings()
    # train_with_users()
    train_with_both('track_user_activations_both')
