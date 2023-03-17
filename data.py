import pandas as pd
from scipy.sparse import csr_array
from sklearn.model_selection import train_test_split

movie_coll_features_path = 'dataset/collaborative_features/movies.csv'
user_coll_features_path = 'dataset/collaborative_features/users.csv'


def read_ratings() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('dataset/ratings.csv')
    train_df, other_df = train_test_split(df, train_size=0.5, random_state=123, shuffle=True)
    dev_df, test_df = train_test_split(other_df, train_size=0.5, random_state=123, shuffle=True)
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
