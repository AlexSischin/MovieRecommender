import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data import read_dev_data
from utils import get_cossim_matrix, argsort2d

file_name = 'dataset/analysis/noisy_data.csv'


def get_sample_data(samples: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev_movie_df, dev_user_df, dev_rating_df, dev_meta_df = read_dev_data(load_meta=True)
    df = pd.concat([dev_meta_df, dev_user_df, dev_movie_df, dev_rating_df], axis=1)

    sample_df = df.sample(samples, random_state=123, ignore_index=True)
    meta_df = sample_df[dev_meta_df.columns]
    movie_df = sample_df[dev_movie_df.columns]
    user_df = sample_df[dev_user_df.columns]
    rating_df = sample_df[dev_rating_df.columns]

    return meta_df, movie_df, user_df, rating_df


def get_difference_matrix(out_arr: np.ndarray) -> np.ndarray:
    diff_map = np.abs(out_arr - out_arr.T)
    masked_diff_map = np.triu(diff_map, 1)
    return masked_diff_map


def main():
    samples = 1000
    top = 1000

    meta_df, movie_df, user_df, rating_df = get_sample_data(samples)

    in_df = pd.concat([movie_df, user_df], axis=1)

    in_df_array = in_df.to_numpy()
    in_df_array[:, 21:41] *= in_df_array[:, 1:21]
    in_df_array = StandardScaler().fit_transform(in_df_array)

    out_df_array = StandardScaler().fit_transform(rating_df.to_numpy())

    sim_matrix = get_cossim_matrix(in_df_array)
    diff_matrix = get_difference_matrix(out_df_array)
    noise_score_matrix = diff_matrix * (sim_matrix + 1) ** 2
    sorted_points = argsort2d(noise_score_matrix)
    top_points = sorted_points[:top]

    showcase_df = pd.DataFrame()
    for match_id, (i, j) in enumerate(top_points):
        cos_sim = sim_matrix[i, j]
        row1 = pd.concat([pd.Series([match_id], index=['match_id']),
                          pd.Series([cos_sim], index=['cosine_similarity']),
                          meta_df.iloc[i],
                          rating_df.iloc[i],
                          movie_df.iloc[i],
                          user_df.iloc[i]])
        row2 = pd.concat([pd.Series([match_id], index=['match_id']),
                          pd.Series([cos_sim], index=['cosine_similarity']),
                          meta_df.iloc[j],
                          rating_df.iloc[j],
                          movie_df.iloc[j],
                          user_df.iloc[j]])
        showcase_df = pd.concat([showcase_df, row1.to_frame().T, row2.to_frame().T])
    showcase_df.set_index('match_id', inplace=True)
    showcase_df.to_csv(file_name)


if __name__ == '__main__':
    main()
