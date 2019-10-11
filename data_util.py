import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from models.double_decoder import MovieSequenceDataset
import torch


def process_movielens_ratings(sequence_length):
    movie_le = LabelEncoder()
    ratings_le = LabelEncoder()
    ratings = pd.read_csv('../data/ml-20m/ratings.csv')
    movies = pd.read_csv('../data/ml-20m/movies.csv')

    movie_le.fit(movies['movieId'].values)
    ratings_le.fit(ratings['rating'].values)

    ratings['rating'] = ratings_le.transform(ratings['rating'])
    ratings['movieId'] = movie_le.transform(ratings['movieId'])

    N = 0
    folder = 'train'
    for key, grp in tqdm(ratings.groupby('userId')):
        grp = grp.sort_values('timestamp')
        for i in range(len(grp) - sequence_length):
            ms = grp['movieId'].values[i: i+sequence_length]
            rs = grp['rating'].values[i: i+sequence_length]
            us = np.array([key] * sequence_length)

            if N <= 10e6:
                folder = 'train'
            elif N > 10e6 and N <= 13e6:
                folder = 'test'
            else:
                folder = 'val'

            with open(f'../data/ml-20m/processed/{folder}/movies/ms_{N}.pickle', 'wb') as f:
                pickle.dump(ms, f)
            with open(f'../data/ml-20m/processed/{folder}/ratings/rs_{N}.pickle', 'wb') as f:
                pickle.dump(rs, f)
            with open(f'../data/ml-20m/processed/{folder}/users/us_{N}.pickle', 'wb') as f:
                pickle.dump(us, f)
            N += 1
