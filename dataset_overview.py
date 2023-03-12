import matplotlib.pyplot as plt
import pandas as pd


def _read(filename):
    print(f'\nFile: {filename}')
    return pd.read_csv(filename)


def check_movies():
    filename = 'dataset/movies.csv'
    print(f'File: {filename}')
    df = pd.read_csv(filename)
    movies = df['movieId']
    print(f'Movies count: {len(movies)}')
    print(f'Movies with unique ID count: {movies.nunique()}')
    print(f'Movies with unique title count: {df["title"].nunique()}')
    print(f'Min movie ID: {min(movies)}')
    print(f'Max movie ID: {max(movies)}')
    genres_series = df['genres'].apply(lambda x: x.split('|'))
    genres = {g for g_list in genres_series for g in g_list}
    print(f'Genres ({len(genres)}): {genres}')


def check_ratings():
    df = _read('dataset/ratings.csv')
    ratings = df['rating'].unique()
    ratings.sort()
    print(f'Unique scores: {ratings}')
    n_unique_movies = df['movieId'].nunique()
    print(f'Number of unique movies: {n_unique_movies}')


def check_tags():
    df = _read('dataset/tags.csv')
    n_unique = df['tag'].str.lower().nunique()
    n_unique_movies = df['movieId'].nunique()
    print(f'Number of unique tags (case insensitive): {n_unique}')
    print(f'Number of unique movies: {n_unique_movies}')


def check_genome_tags():
    df = _read('dataset/genome-tags.csv')
    raw_tags_df = pd.read_csv('dataset/tags.csv')
    unique_tags = raw_tags_df['tag'].str.lower().unique()
    df['presentInRawTags'] = df['tag'].str.lower().isin(unique_tags)
    absent_tags = df[~df['presentInRawTags']]
    print(f'Number of tags absent in ratings.csv (case insensitive): {len(absent_tags)}')


def check_genome_scores():
    df = _read('dataset/genome-scores.csv')
    n_unique_mov = df['movieId'].nunique()
    print(f'Unique movie number: {n_unique_mov}')
    genome_tag_df = pd.read_csv('dataset/genome-tags.csv')
    tag_ids = genome_tag_df['tagId'].unique()
    valid_tag_ids = df['tagId'].isin(tag_ids)
    absent_tag_ids = df[~valid_tag_ids]
    print(f'Number of scores with tag ids absent in genome-tags.csv: {len(absent_tag_ids)}')
    relevance = df['relevance']
    print(f'Min relevance: {min(relevance)}')
    print(f'Max relevance: {max(relevance)}')
    relevance.hist(bins=1000, grid=False)


if __name__ == '__main__':
    check_movies()
    check_ratings()
    check_tags()
    check_genome_tags()
    check_genome_scores()
    plt.show()
