import torch
from sklearn.preprocessing import LabelEncoder
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os, sys
import pickle
import pandas as pd


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + np.sqrt(self.d_model) + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape) * -np.inf, k=1).astype('float32')
    return torch.from_numpy(subsequent_mask)


class MovieSequenceDataset(Dataset):
    def __init__(self, ratings_df: pd.DataFrame, sequence_length, test_pct=0.1):
        self.movie_le = LabelEncoder()
        self.ratings_le = LabelEncoder()
        self.user_le = LabelEncoder()
        
        self.ratings_df = ratings_df.sort_values('timestamp')

        self.movie_le.fit(self.ratings_df['movieId'])
        self.ratings_le.fit(self.ratings_df['rating'])
        self.user_le.fit(self.ratings_df['userId'])

        self.MOVIE_PADDING_IDX = len(self.movie_le.classes_)
        self.MOVIE_CLS = self.MOVIE_PADDING_IDX + 1
        self.MOVIE_SEP = self.MOVIE_PADDING_IDX + 2
        
        self.RATING_PADDING_IDX = len(self.ratings_le.classes_)
        self.RATING_CLS = self.RATING_PADDING_IDX + 1
        self.RATING_SEP = self.RATING_PADDING_IDX + 2
        
        self.USER_PADDING_IDX = len(self.user_le.classes_)
        self.USER_CLS = self.USER_PADDING_IDX + 1
        self.USER_SEP = self.USER_PADDING_IDX + 2
        
        self.ratings_df['movieId'] = self.movie_le.transform(self.ratings_df['movieId'])
        self.ratings_df['rating'] = self.ratings_le.transform(self.ratings_df['rating'])
        self.ratings_df['userId'] = self.user_le.transform(self.ratings_df['userId'])

        self.ratings_df['index'] = range(len(ratings_df))
        self.sequence_length = sequence_length

        self.train_df = self.ratings_df.groupby('userId').apply(lambda x: x.head(int(len(x) * (1 - test_pct)))).reset_index(drop=True)
        self.test_df = self.ratings_df.groupby('userId').apply(lambda x: x.tail(int(len(x) * test_pct))).reset_index(drop=True)

        # Active Data
        self.df = self.train_df
        self.gb = self.train_df.groupby('userId')
        self.users = self.train_df['userId'].values

    def __getitem__(self, index):
        user = self.users[index]
        grp = self.gb.get_group(user)
        grp = grp.loc[grp['index'] <= index]
        grp = grp.tail(self.sequence_length)
        pad = np.ones(max(self.sequence_length - len(grp), 0))

        r = np.append(pad * self.RATING_PADDING_IDX, grp['rating'].values)
        m = np.append(pad * self.MOVIE_PADDING_IDX, grp['movieId'].values)
        u = np.append(pad * self.USER_PADDING_IDX, grp['userId'].values)

        r = np.concatenate(([self.RATING_CLS],
                            r[:self.sequence_length//2],
                            [self.RATING_SEP],
                            r[self.sequence_length//2:],
                            [self.RATING_SEP]))
        
        m = np.concatenate(([self.MOVIE_CLS],
                            m[:self.sequence_length//2],
                            [self.MOVIE_SEP],
                            m[self.sequence_length//2:],
                            [self.MOVIE_SEP]))
        
        u = np.concatenate(([self.USER_CLS],
                            u[:self.sequence_length//2],
                            [self.USER_SEP],
                            u[self.sequence_length//2:],
                            [self.USER_SEP]))
        return torch.LongTensor(m), torch.LongTensor(r), torch.LongTensor(u)

    def __len__(self):
        return len(self.df)

    def train(self):
        self.df = self.train_df
        self.gb = self.train_df.groupby('userId')
        self.users = self.train_df['userId'].values

    def test(self):
        self.df = self.test_df
        self.gb = self.test_df.groupby('userId')
        self.users = self.test_df['userId'].values

# TODO: bigger output networks and multiply by embedding.T
# TODO: Implement swoosh activation for funsies

class MovieSequenceEncoder(torch.nn.Module):
    def __init__(self, sequence_length, n_movies, embedding_dim, n_head, ff_dim, n_encoder_layers, n_ratings=None,
                 model_name='movie_encoder'):
        super().__init__()
        self.model_name = model_name
        self.MOVIE_PADDING_IDX = n_movies
        self.MOVIE_CLS = n_movies + 1

        self.n_ratings = n_ratings
        self.RATING_PADDING_IDX = n_ratings
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        if n_ratings is not None:
            self.RATING_CLS = n_ratings + 1
            self.rating_embedding_layer = torch.nn.Embedding(n_ratings+3, embedding_dim, padding_idx=n_ratings)
            self.masked_rating_output_layer = torch.nn.Linear(embedding_dim, n_ratings)

        self.embedding_layer = torch.nn.Embedding(n_movies+3, embedding_dim, padding_idx=n_movies)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, n_head, ff_dim)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        self.masked_movie_output_layer = torch.nn.Linear(embedding_dim, n_movies)
        self.matching_output_layer = torch.nn.Linear(embedding_dim, 2)

        self.pe = PositionalEncoding(embedding_dim, 0.1)

    def forward(self, *input, **kwargs):
        movies = input[0]
        ratings = kwargs.get('ratings', None)

        movies_embed = self.embedding_layer(movies)
        movies_embed = self.pe(movies_embed)

        if ratings is not None:
            ratings = self.rating_embedding_layer(ratings)
            ratings = self.pe(ratings)
            movies_embed = movies_embed + ratings

        movies_embed = movies_embed.permute(1, 0, 2)
        movies_embed = self.encoder(movies_embed)
        movies_embed = movies_embed.permute(1, 0, 2)
        return movies_embed

    def _gen_random_masks(self, x: torch.LongTensor, pct=0.1):
        """

        :param x: shape (batch_size, sequence_length, 1)
        :param pct:
        :return:
        """
        S = x.shape[1]
        n_masked_idx = int((x.shape[1] - 3) * pct)
        mask = np.append([1] * n_masked_idx, [0] * (x.shape[1] - n_masked_idx - 3))
        mask = np.array([np.random.permutation(mask) for _ in range(x.shape[0])])
        zeros = np.zeros((x.shape[0], 1))
        mask = np.hstack((zeros,
                          mask[:, :S//2],
                          zeros,
                          mask[:, S//2:],
                          zeros))

        mask = torch.BoolTensor(mask)
        return mask

    def pretrain(self, dataloader: DataLoader, **train_kwargs):
        """

        :param movies: [N, seq_length, 1] sequence of movies watched by a user
        :param ratings: [N, seq_length, 1] the corresponding ratings assigned by the user (optional)
        :param train_kwargs:
        :return:
        """
        self.train()
        lr = train_kwargs.get('lr', 1.0e-4)
        epochs = train_kwargs.get('epochs', 1)
        mask_pct = train_kwargs.get('mask_pct', 0.1)
        print_iter = train_kwargs.get('print_iter', 10)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        S = self.sequence_length
        mask_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        matching_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        history = dict(epoch=[], step=[], masked_movie_loss=[], masked_rating_loss=[], matching_loss=[], loss=[],
                       masked_movie_acc=[], masked_rating_acc=[], matching_acc=[])
        for epoch in tqdm(range(epochs)):
            step = 0
            for m, r, u in tqdm(dataloader):

                if self.n_ratings is None:
                    r = None
                opt.zero_grad()

                # Randomly shuffle the second half of the sequences for half the batch
                is_correct_matchup = np.random.choice([0, 1], m.shape[0])
                shuffled_movies = m[is_correct_matchup == 0, (S // 2) + 2: S + 2]
                shuffled_movies = shuffled_movies[torch.randperm(shuffled_movies.size()[0])]
                m[is_correct_matchup == 0, (S // 2) + 2: S + 2] = shuffled_movies
                if r is not None:
                    shuffled_ratings = r[is_correct_matchup == 0, (S // 2) + 2: S + 2]
                    shuffled_ratings = shuffled_ratings[torch.randperm(shuffled_ratings.size()[0])]
                    r[is_correct_matchup == 0, (S // 2) + 2: S + 2] = shuffled_ratings

                m_tgt = m.clone().detach()
                if r is not None:
                    r_tgt = r.clone().detach()

                # Generate masks for random movies
                movie_masks = self._gen_random_masks(m, mask_pct)
                movie_masks[m >= self.MOVIE_PADDING_IDX] = 0
                m[movie_masks] = self.MOVIE_PADDING_IDX
                m = m.cuda()

                if r is not None:
                    ratings_mask = self._gen_random_masks(r, mask_pct)
                    ratings_mask[r >= self.RATING_PADDING_IDX] = 0
                    r[ratings_mask] = self.RATING_PADDING_IDX
                    r = r.cuda()

                out = self.forward(m, ratings=r)  # -> shape (batch_size, sequence_length, embedding_dim)

                # Masked movie predictions
                movies_to_predict = out[movie_masks]
                movie_masked_pred = self.masked_movie_output_layer(movies_to_predict)
                movie_mask_tgt_batch = m_tgt[movie_masks].cuda()
                movie_mask_batch_loss = mask_loss(movie_masked_pred, movie_mask_tgt_batch)

                if r is not None:
                    # Masked rating predictions
                    ratings_to_predict = out[ratings_mask]
                    ratings_masked_pred = self.masked_rating_output_layer(ratings_to_predict)
                    ratings_mask_tgt_batch = r_tgt[ratings_mask].cuda()
                    ratings_mask_batch_loss = mask_loss(ratings_masked_pred, ratings_mask_tgt_batch)
                else:
                    ratings_mask_batch_loss = -1

                is_correct_pred = self.matching_output_layer(out[:, 0, :])
                is_correct_matchup = torch.LongTensor(is_correct_matchup).cuda()
                is_correct_loss = matching_loss(is_correct_pred, is_correct_matchup)
                if r is not None:
                    batch_loss = (movie_mask_batch_loss + ratings_mask_batch_loss + is_correct_loss) / 3.
                else:
                    batch_loss = (movie_mask_batch_loss + is_correct_loss) / 2.

                batch_loss.backward()
                opt.step()

                if step == 0 or (step + 1) % print_iter == 0:
                    movie_batch_acc = (
                            movie_masked_pred.detach().cpu().numpy().argmax(
                                1) == movie_mask_tgt_batch.detach().cpu().numpy()).astype(
                        int).mean()
                    if r is not None:
                        rating_batch_acc = (
                                ratings_masked_pred.detach().cpu().numpy().argmax(
                                    1) == ratings_mask_tgt_batch.detach().cpu().numpy()).astype(
                            int).mean()
                    else:
                        rating_batch_acc = -1
                    top_5_pred = np.argsort(movie_masked_pred.detach().cpu().numpy(), axis=1)[:, -5:]
                    top_5_acc = np.array(
                        [t in p for t, p in zip(movie_mask_tgt_batch.detach().cpu().numpy(), top_5_pred)]).astype(int).mean()
                    matching_acc = (is_correct_pred.detach().cpu().numpy().argmax(
                        1) == is_correct_matchup.detach().cpu().numpy()).astype(int).mean()

                    history['epoch'].append(epoch)
                    history['step'].append(step)
                    history['loss'].append(batch_loss)
                    history['masked_movie_loss'].append(movie_mask_batch_loss)
                    history['masked_movie_acc'].append(movie_batch_acc)
                    history['matching_loss'].append(is_correct_loss)
                    history['matching_acc'].append(matching_acc)
                    if r is not None:
                        history['masked_rating_loss'].append(ratings_mask_batch_loss)
                        history['masked_rating_acc'].append(rating_batch_acc)
                    print(
                        f'Epoch: {epoch}, Step: {step}, Loss: {batch_loss}, Movie Acc:'
                        f'{movie_batch_acc}, Top 5 Acc: {top_5_acc}, Matching Acc: {matching_acc}, Rating Acc: {rating_batch_acc}')
                step += 1
            torch.save(self, f'{self.model_name}_checkpoint_{epoch}.torch')
            with open(f'history_{self.model_name}_{epoch}.pickle', 'wb') as f:
                pickle.dump(history, f)
        with open(f'history_{self.model_name}.pickle', 'wb') as f:
            pickle.dump(history, f)


class MovieLensRecommender(torch.nn.Module):
    def __init__(self, sequence_length, n_users, n_movies, n_ratings, embedding_dim, n_head, ff_dim, n_encoder_layers,
                 n_decoder_layers):
        self.MOVIE_MASK_IDX = n_movies
        self.USER_MASK_IDX = n_users
        self.RATING_MASK_IDX = n_ratings

        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        self.user_embedding = torch.nn.Embedding(n_users + 1, embedding_dim)
        self.movie_embedding = torch.nn.Embedding(n_movies + 1, embedding_dim)
        self.rating_embedding = torch.nn.Embedding(n_ratings, embedding_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, n_head, ff_dim)
        self.user_transformer = torch.nn.Transformer(embedding_dim, n_head, n_encoder_layers, n_decoder_layers, ff_dim)
        self.movie_encoder = torch.nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        self.user_dense_layer = torch.nn.Linear(embedding_dim, n_users)
        self.user_matching_layer = torch.nn.Linear(embedding_dim, 2)

        self.pe = PositionalEncoding(embedding_dim, 0.1)
        # self.user_pe = PositionalEncoding(embedding_dim, 0.1)

    def user_model_forward(self, src, tgt, mask):
        tgt[mask] = self.RATING_MASK_IDX  # Set the masked values to the embedding pad idx

        src = self.user_embedding(src)
        src = src + np.sqrt(self.embedding_dim)
        src = self.pe(src)

        tgt = self.rating_embedding(tgt)
        tgt = tgt + np.sqrt(self.embedding_dim)
        tgt = self.pe(tgt)

        # Encoder expects shape (seq_length, batch_size, embedding_dim)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        # Then we pass it through the encoder stack
        # out = self.encoder(src, src_key_padding_mask=mask, src_mask)
        out = self.user_transformer(src, tgt)
        # Encoder outputs shape (seq_length, batch_size, embedding_dim)
        out = out.permute(1, 0, 2)
        return out

    def pretrain_user_model(self, src, tgt, **train_kwargs):
        lr = train_kwargs.get('lr', 0.001)
        batch_size = train_kwargs.get('batch_size', 512)
        steps = train_kwargs.get('steps', 100)
        mask_pct = train_kwargs.get('mask_pct', 0.1)
        print_iter = train_kwargs.get('print_iter', 100)
        save_iter = train_kwargs.get('save_iter', 100000)
        model_name = train_kwargs.get('model_name', 'user_model_pretrain')

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        N = src.shape[0]
        mask_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        matching_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        for step in tqdm(range(steps)):
            opt.zero_grad()
            idxs = np.random.choice(N, batch_size)

            # Sample a batch of user sequences for movies and the corresponding ratings
            src_batch, tgt_batch = src[idxs], tgt[idxs]

            # Randomly shuffle the second half of the sequences for half the batch
            is_correct_matchup = np.random.choice([0, 1], batch_size)
            shuffled_lineups = src_batch[is_correct_matchup == 0, (self.sequence_length // 2) + 2: self.sequence_length + 2]
            shuffled_lineups = shuffled_lineups[torch.randperm(shuffled_lineups.size()[0])]
            src_batch[is_correct_matchup == 0, (self.sequence_length // 2) + 2: self.sequence_length + 2] = shuffled_lineups

            # Generate masks for random tokens
            masks = self._gen_random_masks(src_batch, mask_pct)

            src_batch = src_batch.cuda()
            tgt_batch = tgt_batch.cuda()
            masks = masks.cuda()

            out = self.user_model_forward(src_batch, tgt_batch, masks)  # -> shape (batch_size, sequence_length, embedding_dim)
            to_predict = out[masks]
            mask_pred = self.user_dense_layer(to_predict)
            mask_tgt_batch = tgt_batch[masks]
            mask_batch_loss = mask_loss(mask_pred, mask_tgt_batch)

            is_correct_pred = self.user_matching_layer(out[:, 0, :])
            is_correct_matchup = torch.LongTensor(is_correct_matchup).cuda()
            is_correct_loss = matching_loss(is_correct_pred, is_correct_matchup)
            batch_loss = (mask_batch_loss + is_correct_loss) / 2.
            batch_loss.backward()
            opt.step()

            if step == 0 or (step + 1) % print_iter == 0:
                batch_acc = (
                            mask_pred.detach().cpu().numpy().argmax(1) == mask_tgt_batch.detach().cpu().numpy()).astype(
                    int).mean()
                top_5_pred = np.argsort(mask_pred.detach().cpu().numpy(), axis=1)[:, -5:]
                top_5_acc = np.array(
                    [t in p for t, p in zip(mask_tgt_batch.detach().cpu().numpy(), top_5_pred)]).astype(int).mean()
                matching_acc = (is_correct_pred.detach().cpu().numpy().argmax(
                    1) == is_correct_matchup.detach().cpu().numpy()).astype(int).mean()

                print(
                    f'Step: {step}, Loss: {batch_loss}, Acc: {batch_acc}, Top 5 Acc: {top_5_acc}, Matching Acc: {matching_acc}')
            if (step + 1) % save_iter == 0:
                torch.save(self, f'{model_name}_checkpoint_{step}.torch')

    def forward(self, *args):
        user_history = args[0]  # Sequence of movies user has watched
        movie_history = args[1]  # Sequence of users who have watched movie
        user_mask = args[2]
        movie_mask = args[3]

        # Mask the indexes we're trying to predict
        user_history[user_mask] = self.USER_MASK_IDX
        movie_history[movie_mask] = self.MOVIE_MASK_IDX

        user_history_embedded = self.movie_embedding(user_history)
        user_history_embedded = user_history_embedded + np.sqrt(self.embedding_dim)
        movie_history_embedded = self.user_embedding(movie_history)
        movie_history_embedded = movie_history_embedded + np.sqrt(self.embedding_dim)

        user_history_embedded = self.user_pe(user_history_embedded)
        movie_history_embedded = self.user_pe(movie_history_embedded)

        user_history_embedded = user_history_embedded.permute(1, 0, 2)
        movie_history_embedded = movie_history_embedded.permute(1, 0, 2)

        user_history_embedded = self.user_decoder(user_history_embedded, user_history_embedded, user_mask)
        movie_history_embedded = self.movie_decoder(movie_history_embedded, movie_history_embedded, movie_mask)

        user_history_embedded = user_history_embedded.permute(1, 0, 2)
        movie_history_embedded = movie_history_embedded.permute(1, 0, 2)

        return user_history_embedded, movie_history_embedded

    def fit(self, user_histories, movie_histories):
        user_masks = subsequent_mask(self.sequence_length)
        movie_masks = subsequent_mask(self.sequence_length)