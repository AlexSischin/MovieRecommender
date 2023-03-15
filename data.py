import pandas as pd
from scipy.sparse import csr_array
from sklearn.model_selection import train_test_split


def read_ratings() -> tuple[csr_array, csr_array, csr_array]:
    df = pd.read_csv('dataset/ratings.csv')
    train_df, other_df = train_test_split(df, train_size=0.5, random_state=123, shuffle=True)
    dev_df, test_df = train_test_split(other_df, train_size=0.5, random_state=123, shuffle=True)
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
