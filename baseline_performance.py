import numpy as np
import pandas as pd

from data import read_ratings


def mean_mse(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    mean_rating = train_df['rating'].mean()
    mse = np.mean((dev_df['rating'] - mean_rating) ** 2)
    return mse


def mean_per_movie_mse(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    movie_ratings = train_df[['movieId', 'rating']].groupby(by='movieId').mean()
    movie_ratings.columns = ['mean_rating']
    ratings = dev_df[['movieId', 'rating']].merge(movie_ratings, left_on='movieId', right_index=True)
    mse = np.mean((ratings['rating'] - ratings['mean_rating']) ** 2)
    return mse


def main():
    train_df, dev_df, _ = read_ratings()

    print(f'Target mean MSE: {mean_mse(train_df, dev_df)}')
    print(f'Target mean per movie MSE: {mean_per_movie_mse(train_df, dev_df)}')


if __name__ == '__main__':
    main()
