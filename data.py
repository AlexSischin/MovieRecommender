import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.model_selection import train_test_split

movie_coll_features_path = 'dataset/collaborative_features/movies.csv'
user_coll_features_path = 'dataset/collaborative_features/users.csv'
content_features_path = 'dataset/content_features/ratings.parquet'


def read_ratings(sort=False, drop_irrelevant_cols=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('dataset/ratings.csv')
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


def generate_and_export_content_features():
    rating_train_df, rating_dev_df, rating_test_df = read_ratings(sort=True, drop_irrelevant_cols=True)
    movies_df = read_movies()
    movie_genres_df = encode_movie_genres(movies_df)
    movie_titles_df = get_movie_titles(movies_df)
    movie_mean_rating_df = calc_movie_mean_rating(rating_train_df)
    user_mean_rating_df = calc_user_mean_rating(rating_train_df, movie_genres_df)

    tmp = pd.concat([rating_train_df, rating_dev_df, rating_test_df],
                    keys=['train', 'dev', 'test'], names=['set', 'id'])
    tmp = tmp.reset_index(level='set')
    tmp = tmp.merge(movie_titles_df, left_on='movieId', right_index=True)
    tmp = tmp.merge(movie_mean_rating_df, left_on='movieId', right_index=True)
    tmp = tmp.merge(movie_genres_df, left_on='movieId', right_index=True)
    tmp = tmp.merge(user_mean_rating_df, left_on='userId', right_index=True)
    tmp = tmp.round(2)
    tmp = tmp.sort_values(by=['set', 'movieId', 'userId'])
    united_df = tmp

    united_df.to_parquet(content_features_path, index=False, engine='fastparquet', compression=None)


def read_content_features():
    content_df = pd.read_parquet(content_features_path, engine='pyarrow')
    grouped_content_df = content_df.groupby('set')

    train_df = grouped_content_df.get_group('train')
    dev_df = grouped_content_df.get_group('dev')
    test_df = grouped_content_df.get_group('test')

    return train_df, dev_df, test_df
