from models.double_decoder import MovieSequenceEncoder, MovieSequenceDataset
from data_util import process_movielens_ratings
import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    PCT_OF_DATA = 0.2
    SEQUENCE_LENGTH = [50]
    ratings = pd.read_csv('../data/ml-20m/ratings.csv')
    np.random.seed(42)
    idxs = np.random.choice(range(len(ratings)), int(len(ratings) * PCT_OF_DATA), replace=False)
    ratings = ratings.iloc[idxs, :]
    embedding_dim = 256
    n_head = 8
    ff_dim = 1024
    n_encoder_layers = 4

    # for sl in SEQUENCE_LENGTH:
    #     dataset = MovieSequenceDataset(ratings, sequence_length=sl, test_pct=0)
    #     dataset.train()
    #
    #     model: MovieSequenceEncoder = MovieSequenceEncoder(sequence_length=sl,
    #                                                        n_movies=len(dataset.movie_le.classes_),
    #                                                        embedding_dim=embedding_dim, n_head=n_head, ff_dim=ff_dim,
    #                                                        n_encoder_layers=n_encoder_layers,
    #                                                        n_ratings=None,
    #                                                        model_name=f'movie_pretrain_no_ratings_{sl}')
    #     print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    #     # 1) Train without ratings as a baseline
    #     batch_size = 512
    #     dl = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    #     model.cuda()
    #     model.pretrain(dl, lr=1.0e-4, print_iter=100, epochs=10, mask_pct=0.5)
    #     torch.save(model, f'{model.model_name}.torch')

    for sl in SEQUENCE_LENGTH:
        dataset = MovieSequenceDataset(ratings, sequence_length=sl, test_pct=0)
        dataset.train()

        model: MovieSequenceEncoder = MovieSequenceEncoder(sequence_length=sl,
                                                           n_movies=len(dataset.movie_le.classes_),
                                                           embedding_dim=embedding_dim, n_head=n_head, ff_dim=ff_dim,
                                                           n_encoder_layers=n_encoder_layers,
                                                           n_ratings=len(dataset.ratings_le.classes_),
                                                           model_name=f'movie_pretrain_with_ratings_{sl}')
        print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        # 1) Train without ratings as a baseline
        batch_size = 512
        dl = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
        model.cuda()
        model.pretrain(dl, lr=1.0e-4, print_iter=100, epochs=10, mask_pct=0.1)
        torch.save(model, f'{model.model_name}.torch')