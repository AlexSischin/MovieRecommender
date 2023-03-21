import numpy as np
import pandas as pd
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
