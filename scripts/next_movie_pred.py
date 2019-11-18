import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from models.double_decoder import MovieSequenceEncoder


def baseline_test():
    baseline_model = torch.load('baseline.torch')
    baseline_model.requires_grad = False
    h = baseline_model.next_movie_prediction(dataloader)
    with open('baseline_test_results.pickle', 'wb') as f:
        pickle.dump(h, f)


def ratings_test():
    model = torch.load('ratings.torch')
    model.requires_grad = False
    h = model.next_movie_prediction(dataloader)
    with open('ratings_test_results.pickle', 'wb') as f:
        pickle.dump(h, f)


def users_test():
    model = torch.load('users.torch')
    model.requires_grad = False
    h = model.next_movie_prediction(dataloader)
    with open('users_test_results.pickle', 'wb') as f:
        pickle.dump(h, f)


def both_test():
    model = torch.load('both.torch')
    model.requires_grad = False
    h = model.next_movie_prediction(dataloader)
    with open('both_test_results.pickle', 'wb') as f:
        pickle.dump(h, f)


if __name__ == '__main__':
    dataset = torch.load('dataset.torch')
    dataset.test()
    dataloader = DataLoader(dataset, batch_size=512, num_workers=2, shuffle=False)
    # baseline_test()
    # ratings_test()
    # users_test()
    both_test()