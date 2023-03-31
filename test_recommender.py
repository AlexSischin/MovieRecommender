import numpy as np
import pandas as pd
from tensorflow import keras as ks

from sklearn.preprocessing import StandardScaler

from data import read_train_data, generate_custom_user

all_my_ratings = [
    ('Terminator 2', 589, 4.0),
    ('Ted', 95441, 0.5),
    ('Matrix', 2571, 3.0),
    ('Avatar', 72998, 2.0),
    ('Spider-man', 5349, 1.0),
    ('Agora', 74624, 3.5),
    ('American History X', 2329, 4.5),
    ('Rush hour', 2273, 4.5),
    ('Interstellar', 109487, 3.5),
    ('No Country for Old Men', 55820, 5.0),
]


def main():
    train_movie_df, train_user_df, _, train_meta_df = read_train_data(load_meta=True)
    train_movie_df = train_movie_df[:1_000_000]
    train_user_df = train_user_df[:1_000_000]
    train_meta_df = train_meta_df[:1_000_000]

    movie_scaler = StandardScaler().fit(train_movie_df.to_numpy())
    user_scaler = StandardScaler().fit(train_user_df.to_numpy())

    movie_df = pd.concat([train_meta_df, train_movie_df], axis=1).drop_duplicates(subset=['movieId'])

    titles = movie_df['title'].to_numpy()
    movies = movie_scaler.transform(movie_df.drop(columns=['userId', 'movieId', 'title']).to_numpy())
    model = ks.models.load_model('model/m7')

    all_user_ratings_df = pd.DataFrame(all_my_ratings, columns=['title', 'movieId', 'rating'])
    for n in range(0, 10):
        ratings = all_user_ratings_df[n:n + 1]
        print(f'Watched movies ({n + 1}): {ratings["title"].to_numpy()}')

        user = generate_custom_user(ratings)
        user = user_scaler.transform(user.to_numpy().reshape(1, -1))
        user = np.repeat(user, movies.shape[0], axis=0)
        user = user * 20

        predictions = model.predict({'movie': movies, 'user': user})
        top_prediction_ids = np.argsort(-predictions.flatten())[:10]
        top_movies = titles[top_prediction_ids]
        print(f'Recommendations: {top_movies}')


if __name__ == '__main__':
    main()
