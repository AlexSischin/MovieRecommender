import csv
import os
import string

import nltk
import numpy as np
import pandas as pd
from pandas.io.pytables import TableIterator
from scipy.sparse import csr_array
from sklearn.model_selection import train_test_split

movie_coll_features_path = 'dataset/collaborative_features/movies.csv'
user_coll_features_path = 'dataset/collaborative_features/users.csv'

train_movie_features_path = 'dataset/content_features/train/movies.parquet'
train_user_features_path = 'dataset/content_features/train/users.parquet'
train_rating_path = 'dataset/content_features/train/ratings.parquet'
train_meta_path = 'dataset/content_features/train/meta.parquet'

dev_movie_features_path = 'dataset/content_features/dev/movies.parquet'
dev_user_features_path = 'dataset/content_features/dev/users.parquet'
dev_rating_path = 'dataset/content_features/dev/ratings.parquet'
dev_meta_path = 'dataset/content_features/dev/meta.parquet'

test_movie_features_path = 'dataset/content_features/test/movies.parquet'
test_user_features_path = 'dataset/content_features/test/users.parquet'
test_rating_path = 'dataset/content_features/test/ratings.parquet'
test_meta_path = 'dataset/content_features/test/meta.parquet'

preprocessed_tags_path = 'dataset/tags/tags.csv'

movie_emb25_path = 'dataset/embedding_features/movie25.parquet'
movie_emb50_path = 'dataset/embedding_features/movie50.parquet'
movie_emb100_path = 'dataset/embedding_features/movie100.parquet'
movie_emb200_path = 'dataset/embedding_features/movie200.parquet'

user_emb25_path = 'dataset/embedding_features/user25.parquet'
user_emb50_path = 'dataset/embedding_features/user50.parquet'
user_emb100_path = 'dataset/embedding_features/user100.parquet'
user_emb200_path = 'dataset/embedding_features/user200.parquet'

embedding_data_path = 'dataset/embedding_features/rating.h5'

MOVIE_FEATURES = ['mean_rating', 'genre_(no genres listed)', 'genre_Action', 'genre_Adventure', 'genre_Animation',
                  'genre_Children', 'genre_Comedy', 'genre_Crime', 'genre_Documentary', 'genre_Drama', 'genre_Fantasy',
                  'genre_Film-Noir', 'genre_Horror', 'genre_IMAX', 'genre_Musical', 'genre_Mystery', 'genre_Romance',
                  'genre_Sci-Fi', 'genre_Thriller', 'genre_War', 'genre_Western']

USER_FEATURES = ['genre_(no genres listed)_mean_rating', 'genre_Action_mean_rating', 'genre_Adventure_mean_rating',
                 'genre_Animation_mean_rating', 'genre_Children_mean_rating', 'genre_Comedy_mean_rating',
                 'genre_Crime_mean_rating', 'genre_Documentary_mean_rating', 'genre_Drama_mean_rating',
                 'genre_Fantasy_mean_rating', 'genre_Film-Noir_mean_rating', 'genre_Horror_mean_rating',
                 'genre_IMAX_mean_rating', 'genre_Musical_mean_rating', 'genre_Mystery_mean_rating',
                 'genre_Romance_mean_rating', 'genre_Sci-Fi_mean_rating', 'genre_Thriller_mean_rating',
                 'genre_War_mean_rating', 'genre_Western_mean_rating']

META_FEATURES = ['movieId', 'title', 'userId']

RATING_FEATURE = ['rating']


def read_ratings(sort=False, drop_irrelevant_cols=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('dataset/ratings.csv')
    df['rating'] = df['rating'].astype(np.float32)
    if drop_irrelevant_cols:
        df.drop(columns=['timestamp'], inplace=True)
    train_df, other_df = train_test_split(df, train_size=0.5, random_state=123, shuffle=True)
    dev_df, test_df = train_test_split(other_df, train_size=0.5, random_state=123, shuffle=True)
    if sort:
        train_df.sort_index(inplace=True)
        dev_df.sort_index(inplace=True)
        test_df.sort_index(inplace=True)
    return train_df, dev_df, test_df


def read_ratings_as_sparse_matrix() -> tuple[csr_array, csr_array, csr_array]:
    train_df, dev_df, test_df = read_ratings()
    train_row, train_col, train_val = train_df['movieId'], train_df['userId'], train_df['rating']
    dev_row, dev_col, dev_val = dev_df['movieId'], dev_df['userId'], dev_df['rating']
    test_row, test_col, test_val = test_df['movieId'], test_df['userId'], test_df['rating']
    train_set = csr_array((train_val, (train_row, train_col)))
    dev_set = csr_array((dev_val, (dev_row, dev_col)))
    test_set = csr_array((test_val, (test_row, test_col)))
    movie_shape = max([s.shape[0] for s in [train_set, dev_set, test_set]])
    user_shape = max([s.shape[1] for s in [train_set, dev_set, test_set]])
    train_set.resize(movie_shape, user_shape)
    dev_set.resize(movie_shape, user_shape)
    test_set.resize(movie_shape, user_shape)
    return train_set, dev_set, test_set


def export_collaborative_features(X, W, b):
    movie_df = pd.DataFrame(X)
    movie_df.columns = [f'f_{i}' for i in range(X.shape[1])]
    movie_df.to_csv(movie_coll_features_path, index=False)

    user_w_df = pd.DataFrame(W)
    user_w_df.columns = [f'w_{i}' for i in range(W.shape[1])]
    user_b_df = pd.DataFrame(b)
    user_b_df.columns = ['b']
    user_df = pd.concat([user_w_df, user_b_df])
    user_df.to_csv(user_coll_features_path, index=False)


def read_movies():
    return pd.read_csv('dataset/movies.csv')


def get_movie_titles(movies_df: pd.DataFrame):
    return movies_df[['movieId', 'title']].set_index(['movieId'])


def encode_movie_genres(movies_df: pd.DataFrame):
    genres_lists_series = movies_df['genres'].apply(lambda g: g.split('|'))
    genres_df = pd.get_dummies(genres_lists_series.explode()).groupby(level=0).sum().add_prefix('genre_')
    movies_df = movies_df.merge(genres_df, left_index=True, right_index=True)
    movies_df.drop(columns=['title', 'genres'], inplace=True)
    movies_df.set_index(['movieId'], inplace=True)
    return movies_df


def calc_movie_mean_rating(rating_df: pd.DataFrame):
    filtered_rating_df = rating_df[['movieId', 'rating']]
    mean_rating_df = filtered_rating_df.groupby(by='movieId').mean(numeric_only=True)
    mean_rating_df.columns = ['mean_rating']
    return mean_rating_df


def calc_user_mean_rating(rating_df: pd.DataFrame, movie_genres_df: pd.DataFrame):
    genre_columns = movie_genres_df.columns
    filtered_rating_df = rating_df[['userId', 'movieId', 'rating']]
    genre_rating_df = filtered_rating_df.merge(movie_genres_df, left_on='movieId', right_index=True)
    genre_rating_df.replace(0, np.NaN, inplace=True)
    genre_rating_df[genre_columns] = genre_rating_df[genre_columns].mul(genre_rating_df['rating'], axis=0)
    genre_rating_df.drop(columns=['movieId', 'rating'], inplace=True)
    mean_rating_df = genre_rating_df.groupby(by='userId').mean(numeric_only=True)
    mean_rating_df.columns = [f'{c}_mean_rating' for c in genre_columns]
    mean_rating = rating_df['rating'].mean()
    mean_rating_df.replace(np.NaN, mean_rating, inplace=True)
    return mean_rating_df


def combine_features(df, movie_titles_df, movie_mean_rating_df, movie_genres_df, user_mean_rating_df):
    df = df.merge(movie_titles_df, left_on='movieId', right_index=True)
    df = df.merge(movie_mean_rating_df, left_on='movieId', right_index=True)
    df = df.merge(movie_genres_df, left_on='movieId', right_index=True)
    df = df.merge(user_mean_rating_df, left_on='userId', right_index=True)
    df = df.sort_values(by=['userId', 'movieId'])
    return df


def export_to_parquet(data, path):
    data.to_parquet(path, index=False, engine='fastparquet', compression=None)


def generate_and_export_content_features():
    rating_train_df, rating_dev_df, rating_test_df = read_ratings(sort=True, drop_irrelevant_cols=True)
    movies_df = read_movies()
    genres_df = encode_movie_genres(movies_df)
    titles_df = get_movie_titles(movies_df)
    mean_rating_df = calc_movie_mean_rating(rating_train_df)
    user_mean_rating_df = calc_user_mean_rating(rating_train_df, genres_df)

    train_df = combine_features(rating_train_df, titles_df, mean_rating_df, genres_df, user_mean_rating_df)
    dev_df = combine_features(rating_dev_df, titles_df, mean_rating_df, genres_df, user_mean_rating_df)
    test_df = combine_features(rating_test_df, titles_df, mean_rating_df, genres_df, user_mean_rating_df)

    input_data_cols = MOVIE_FEATURES + USER_FEATURES + RATING_FEATURE
    train_df[input_data_cols] = train_df[input_data_cols].astype(np.float32).round(2)
    dev_df[input_data_cols] = dev_df[input_data_cols].astype(np.float32).round(2)
    test_df[input_data_cols] = test_df[input_data_cols].astype(np.float32).round(2)

    export_to_parquet(train_df[MOVIE_FEATURES], train_movie_features_path)
    export_to_parquet(train_df[USER_FEATURES], train_user_features_path)
    export_to_parquet(train_df[RATING_FEATURE], train_rating_path)
    export_to_parquet(train_df[META_FEATURES], train_meta_path)

    export_to_parquet(dev_df[MOVIE_FEATURES], dev_movie_features_path)
    export_to_parquet(dev_df[USER_FEATURES], dev_user_features_path)
    export_to_parquet(dev_df[RATING_FEATURE], dev_rating_path)
    export_to_parquet(dev_df[META_FEATURES], dev_meta_path)

    export_to_parquet(test_df[MOVIE_FEATURES], test_movie_features_path)
    export_to_parquet(test_df[USER_FEATURES], test_user_features_path)
    export_to_parquet(test_df[RATING_FEATURE], test_rating_path)
    export_to_parquet(test_df[META_FEATURES], test_meta_path)


def read_parquet(file, engine='pyarrow'):
    return pd.read_parquet(file, engine=engine)


def read_train_data(load_meta=False):
    movie_df = read_parquet(train_movie_features_path)
    user_df = read_parquet(train_user_features_path)
    rating_df = read_parquet(train_rating_path)
    meta_df = read_parquet(train_meta_path) if load_meta else None
    return movie_df, user_df, rating_df, meta_df


def read_dev_data(load_meta=False):
    movie_df = read_parquet(dev_movie_features_path)
    user_df = read_parquet(dev_user_features_path)
    rating_df = read_parquet(dev_rating_path)
    meta_df = read_parquet(dev_meta_path) if load_meta else None
    return movie_df, user_df, rating_df, meta_df


def read_test_data(load_meta=False):
    movie_df = read_parquet(test_movie_features_path)
    user_df = read_parquet(test_user_features_path)
    rating_df = read_parquet(test_rating_path)
    meta_df = read_parquet(test_meta_path) if load_meta else None
    return movie_df, user_df, rating_df, meta_df


def read_tags(drop_irrelevant_cols=True, drop_na=True):
    df = pd.read_csv('dataset/tags.csv')
    if drop_irrelevant_cols:
        df.drop(columns=['userId', 'timestamp'], inplace=True)
    if drop_na:
        df.dropna(inplace=True)
    return df


def read_genome_scores(drop_na=True):
    score_df = pd.read_csv('dataset/genome-scores.csv')
    id_df = pd.read_csv('dataset/genome-tags.csv')
    if drop_na:
        score_df.dropna(inplace=True)
        id_df.dropna(inplace=True)
    return score_df, id_df


def normalize_text(text: pd.Series, remove_digits=False):
    init_text = text
    text = text.str.lower()
    if remove_digits:
        text = text.replace(r'\d', '', regex=True)
    text = text.replace(r'\n', '', regex=True)
    text = text.replace('-', '')
    text = text.replace(f'[{string.punctuation}]', '', regex=True)
    text = text.str.split()
    text = text.explode()
    text.dropna(inplace=True)
    text = text[~text.isin(nltk.corpus.stopwords.words("english"))]
    text = text.groupby(level=0).agg(' '.join)
    text = pd.merge(init_text, text, how='left', left_index=True, right_index=True, suffixes=('_', ''))[text.name]
    return text


def normalize_genome_score(score):
    s = lambda x: 1 / (1 + np.e ** -x)
    return 2 * s(score * 2) - 1


def process_and_export_tag_info():
    nltk.download('stopwords')

    print('Reading and preparing data')
    tag_df = read_tags()
    score_df, tag_id_df = read_genome_scores()
    title_df = read_movies()
    score_df['relevance'] = score_df['relevance']
    mean_relevance = score_df['relevance'].mean()

    print('Normalizing tags')
    tag_df['tag'] = normalize_text(tag_df['tag'])
    tag_id_df['tag'] = normalize_text(tag_id_df['tag'])
    title_df['tag'] = normalize_text(title_df['title'], remove_digits=True)

    print('Dropping empty tags')
    tag_df.dropna(inplace=True)
    tag_id_df.dropna(inplace=True)
    title_df.dropna(inplace=True)

    print('Merging scores')
    score_df = score_df.merge(tag_id_df, on='tagId')[['movieId', 'tag', 'relevance']]
    tag_df = tag_df.merge(score_df, how='outer', on=['movieId', 'tag'])
    tag_df['relevance'] = tag_df['relevance'].fillna(mean_relevance)

    print('Merging title tags')
    tag_df = pd.concat([tag_df, title_df[['movieId', 'tag']]])
    tag_df['relevance'] = tag_df['relevance'].fillna(1)

    print('Computing relevance score')
    tag_df = tag_df.groupby(by=['movieId', 'tag']).sum().reset_index()
    tag_df['relevance'] = normalize_genome_score(tag_df['relevance']).astype(np.float32)

    print('Exporting')
    export_to_parquet(tag_df, preprocessed_tags_path)


def read_embeddings(filepath):
    df = pd.read_csv(filepath, sep=' ', header=None, quoting=csv.QUOTE_NONE)
    df.columns = df.columns.astype(str)
    df = df.set_index('0')
    df = df[~df.index.duplicated(keep=False)]
    df = df.astype(np.float32)
    return df


def get_df_norm(df: pd.DataFrame):
    return np.sqrt(np.square(df).sum(axis=1))


def generate_movie_embeddings(tag_df: pd.DataFrame, emb_df: pd.DataFrame) -> pd.DataFrame:
    tag_df = tag_df.copy()
    tag_df = tag_df.merge(emb_df, how='inner', left_on='tagWord', right_index=True)
    tag_df[emb_df.columns] = tag_df[emb_df.columns].mul(tag_df['relevance'], axis=0)
    tag_df.drop(columns=['tagWord', 'relevance'], inplace=True)
    tag_df = tag_df.groupby(['tagId', 'movieId']).mean()
    tag_df.reset_index(level=0, drop=True, inplace=True)
    tag_df = tag_df.groupby(level=0).sum()
    norm = get_df_norm(tag_df)
    tag_df = tag_df.divide(norm, axis=0)
    tag_df.reset_index(names='movieId', inplace=True)
    return tag_df.copy()


def generate_and_export_movie_embeddings():
    print('Reading data from disk')
    tag_df = read_parquet(preprocessed_tags_path)
    emb25_df = read_embeddings('dataset/glove_twitter/glove.twitter.27B.25d.txt')
    emb50_df = read_embeddings('dataset/glove_twitter/glove.twitter.27B.50d.txt')
    emb100_df = read_embeddings('dataset/glove_twitter/glove.twitter.27B.100d.txt')
    emb200_df = read_embeddings('dataset/glove_twitter/glove.twitter.27B.200d.txt')

    print('Preprocessing tags')
    tag_df['tagWord'] = tag_df['tag'].str.split()
    tag_df.drop(columns=['tag'], inplace=True)
    tag_df = tag_df.explode('tagWord')
    tag_df.reset_index(inplace=True, names='tagId')

    print('Preprocessing data for iteration')
    mov25_df = pd.DataFrame()
    mov50_df = pd.DataFrame()
    mov100_df = pd.DataFrame()
    mov200_df = pd.DataFrame()
    movie_ids = tag_df['movieId']
    chunk_count = 12
    movie_chunk_ids = np.array_split(movie_ids.unique(), chunk_count)
    for i, chunk_ids in enumerate(movie_chunk_ids):
        print(f'Iteration {i + 1} of {chunk_count}'.center(98, '-'))
        print('Selecting tags')
        tag_sub_df = tag_df[movie_ids.isin(chunk_ids)]
        print('Generating movie embeddings')
        mov25_sub_df = generate_movie_embeddings(tag_sub_df, emb25_df)
        mov50_sub_df = generate_movie_embeddings(tag_sub_df, emb50_df)
        mov100_sub_df = generate_movie_embeddings(tag_sub_df, emb100_df)
        mov200_sub_df = generate_movie_embeddings(tag_sub_df, emb200_df)
        print('Concatenating embeddings')
        mov25_df = pd.concat([mov25_df, mov25_sub_df])
        mov50_df = pd.concat([mov50_df, mov50_sub_df])
        mov100_df = pd.concat([mov100_df, mov100_sub_df])
        mov200_df = pd.concat([mov200_df, mov200_sub_df])

    print('Exporting embeddings')
    export_to_parquet(mov25_df, movie_emb25_path)
    export_to_parquet(mov50_df, movie_emb50_path)
    export_to_parquet(mov100_df, movie_emb100_path)
    export_to_parquet(mov200_df, movie_emb200_path)


def generate_user_embeddings(rating_df: pd.DataFrame, movie_df: pd.DataFrame) -> pd.DataFrame:
    user_df = rating_df.merge(movie_df, how='inner', on='movieId')

    user_df.set_index('userId', inplace=True)
    relevance_scores = user_df['rating']
    user_df.drop(columns=['movieId', 'rating'], inplace=True)
    user_df = user_df.mul(relevance_scores, axis=0)

    user_df = user_df.groupby(level=0).sum()
    norm = get_df_norm(user_df)
    user_df = user_df.divide(norm, axis=0)
    user_df = user_df.reset_index()

    return user_df.copy()


def generate_and_export_user_embeddings():
    print('Reading from disk')
    rating_df, _, _ = read_ratings(sort=True, drop_irrelevant_cols=True)
    mov25_df = read_parquet(movie_emb25_path)
    mov50_df = read_parquet(movie_emb50_path)
    mov100_df = read_parquet(movie_emb100_path)
    mov200_df = read_parquet(movie_emb200_path)

    usr25_df = pd.DataFrame()
    usr50_df = pd.DataFrame()
    usr100_df = pd.DataFrame()
    usr200_df = pd.DataFrame()
    batch_count = 1
    user_ids = rating_df['userId']
    user_ids_batches = np.array_split(user_ids.unique(), batch_count)
    for i, user_ids_batch in enumerate(user_ids_batches):
        print(f'Iteration {i + 1} of {batch_count}'.center(98, '-'))
        print('Selecting ratings')
        rating_sub_df = rating_df[user_ids.isin(user_ids_batch)]
        print('Generating movie embeddings')
        usr25_sub_df = generate_user_embeddings(rating_sub_df, mov25_df)
        usr50_sub_df = generate_user_embeddings(rating_sub_df, mov50_df)
        usr100_sub_df = generate_user_embeddings(rating_sub_df, mov100_df)
        usr200_sub_df = generate_user_embeddings(rating_sub_df, mov200_df)
        print('Concatenating embeddings')
        usr25_df = pd.concat([usr25_df, usr25_sub_df])
        usr50_df = pd.concat([usr50_df, usr50_sub_df])
        usr100_df = pd.concat([usr100_df, usr100_sub_df])
        usr200_df = pd.concat([usr200_df, usr200_sub_df])

    print('Exporting embeddings')
    export_to_parquet(usr25_df, user_emb25_path)
    export_to_parquet(usr50_df, user_emb50_path)
    export_to_parquet(usr100_df, user_emb100_path)
    export_to_parquet(usr200_df, user_emb200_path)


def generate_embedding_data(ratings_df: pd.DataFrame, mov_emb_df: pd.DataFrame, usr_emb_df: pd.DataFrame,
                            include_unknown=True):
    mov_emb_df = mov_emb_df.copy().set_index('movieId')
    mov_emb_df.columns = [f'm_{c}' for c in mov_emb_df]
    usr_emb_df = usr_emb_df.copy().set_index('userId')
    usr_emb_df.columns = [f'u_{c}' for c in usr_emb_df]
    if include_unknown:
        mean_movie = mov_emb_df.mean()
        mean_movie = mean_movie.divide(np.linalg.norm(mean_movie), axis=0).squeeze()
        mean_user = usr_emb_df.mean()
        mean_user = mean_user.divide(np.linalg.norm(mean_user), axis=0).squeeze()
        df = pd.merge(ratings_df, mov_emb_df, left_on='movieId', right_index=True, how='left')
        df = pd.merge(df, usr_emb_df, left_on='userId', right_index=True, how='left')
        df.fillna(mean_movie, inplace=True)
        df.fillna(mean_user, inplace=True)
    else:
        df = pd.merge(ratings_df, mov_emb_df, left_on='movieId', right_index=True, how='inner')
        df = pd.merge(df, usr_emb_df, left_on='movieId', right_index=True, how='inner')
    df.drop(columns=['movieId', 'userId'], inplace=True)
    return df


def generate_and_export_embedding_data():
    print('Reading from disk')
    rating_train_df, rating_dev_df, rating_test_df = read_ratings(sort=True, drop_irrelevant_cols=True)
    mov25_df = read_parquet(movie_emb25_path)
    mov50_df = read_parquet(movie_emb50_path)
    mov100_df = read_parquet(movie_emb100_path)
    mov200_df = read_parquet(movie_emb200_path)
    usr25_df = read_parquet(user_emb25_path)
    usr50_df = read_parquet(user_emb50_path)
    usr100_df = read_parquet(user_emb100_path)
    usr200_df = read_parquet(user_emb200_path)

    print('Preparing for generation')
    rating_df_map = {
        'train': np.array_split(rating_train_df, 12, axis=0),
        'dev': np.array_split(rating_dev_df, 6, axis=0),
        'test': np.array_split(rating_test_df, 6, axis=0)
    }
    embedding_map = {
        '25': (mov25_df, usr25_df),
        '50': (mov50_df, usr50_df),
        '100': (mov100_df, usr100_df),
        '200': (mov200_df, usr200_df),
    }
    if os.path.exists(embedding_data_path):
        os.remove(embedding_data_path)
    with pd.HDFStore(embedding_data_path) as store:
        print('Generating and exporting embeddings')
        for emb_size, embeddings in embedding_map.items():
            print(f'Embeddings: {emb_size}d'.center(98, '-'))
            mov_df, usr_df = embeddings
            for set_name, rating_sub_dfs in rating_df_map.items():
                print(f'Set: {set_name}'.center(98, '-'))
                key = f'/d{emb_size}/{set_name}'
                for i, rating_sub_df in enumerate(rating_sub_dfs):
                    print(f'It: {i + 1} / {len(rating_sub_dfs)}'.center(98, '-'))
                    include_unknown = set_name != 'train'
                    df = generate_embedding_data(rating_sub_df, mov_df, usr_df, include_unknown=include_unknown)
                    store.append(key, df)


def load_emb_data(dim: int, train_chunk_size: int, dev_chunk_size: int, test_chunk_size: int
                  ) -> tuple[TableIterator, TableIterator, TableIterator]:
    train_df_it = pd.read_hdf(embedding_data_path, f'/d{dim}/train', chunksize=train_chunk_size)
    dev_df_it = pd.read_hdf(embedding_data_path, f'/d{dim}/dev', chunksize=dev_chunk_size)
    test_df_it = pd.read_hdf(embedding_data_path, f'/d{dim}/dev', chunksize=test_chunk_size)
    return train_df_it, dev_df_it, test_df_it


def load_samples(dim: int, train_sample_size: int, dev_sample_size: int, test_sample_size: int
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with pd.HDFStore(embedding_data_path) as store:
        train_key, dev_key, test_key = f'/d{dim}/train', f'/d{dim}/dev', f'/d{dim}/dev'
        train_ids = np.random.randint(0, store.get_storer(train_key).nrows, size=train_sample_size)
        dev_ids = np.random.randint(0, store.get_storer(dev_key).nrows, size=dev_sample_size)
        test_ids = np.random.randint(0, store.get_storer(test_key).nrows, size=test_sample_size)
        train_sample_df = pd.read_hdf(store, train_key, where=train_ids)
        def_sample_df = pd.read_hdf(store, dev_key, where=dev_ids)
        test_sample_df = pd.read_hdf(store, test_key, where=test_ids)
    return train_sample_df, def_sample_df, test_sample_df
