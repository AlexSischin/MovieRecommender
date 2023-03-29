import re

import numpy as np
import pandas as pd

from data import load_samples, read_movies, read_tags
from utils import get_cossim_matrix, argsort2d

similar_movies_path = 'dataset/analysis/similar_movie_embeddings.csv'


def main():
    np.random.seed(0)
    train_df, _, _ = load_samples(200, 5_000, 2_500, 2_500)
    movies_df = read_movies()
    tags_df = read_tags()

    train_df.drop_duplicates(subset='movieId', inplace=True)
    train_df.sort_values(by='movieId', inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    tags_df.drop_duplicates(subset=['movieId', 'tag'], inplace=True)
    tags_df = tags_df.sort_values('tag')
    tags_df = tags_df.groupby('movieId').agg(' | '.join)
    tags_df.rename(columns={'tag': 'tags'}, inplace=True)
    tags_df.reset_index(inplace=True)

    movie_columns = [c for c in train_df if re.match(r'm_\d+', c)]
    movie_ids = train_df['movieId']
    movie_emb_df = train_df[movie_columns]

    cossim_matrix = get_cossim_matrix(movie_emb_df.to_numpy())
    best_match_ids = argsort2d(cossim_matrix)

    match_ids = np.arange(0, best_match_ids.shape[0]).repeat(2)
    cos_similarities = cossim_matrix[best_match_ids.T[0], best_match_ids.T[1]].repeat(2)
    sample_ids = best_match_ids.flatten()
    sim_movies_df = pd.DataFrame(
        np.stack([match_ids, cos_similarities, sample_ids]).T,
        columns=['matchId', 'cosineSimilarity', 'sampleId'])
    sim_movies_df.dropna(inplace=True)
    sim_movies_df['matchId'] = sim_movies_df['matchId'].astype(np.int64)
    sim_movies_df = sim_movies_df.merge(movie_ids, how='left', left_on='sampleId', right_index=True)
    sim_movies_df = sim_movies_df.merge(movies_df, how='left', on='movieId')
    sim_movies_df = sim_movies_df.merge(tags_df, how='left', on='movieId')
    sim_movies_df = sim_movies_df.sort_values(by='matchId').reset_index(drop=True)
    sim_movies_df = sim_movies_df.drop(columns=['sampleId'])
    sim_movies_df.to_csv(similar_movies_path, index=False)


if __name__ == '__main__':
    main()
