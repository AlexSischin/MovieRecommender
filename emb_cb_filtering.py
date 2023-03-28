import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as ks

from data import load_samples
from utils import compare_history, compare_distributions

model_architecture_path = 'report/'


def prepare_data(df: pd.DataFrame):
    columns = df.columns.tolist()
    movie_columns = [c for c in columns if re.match(r'm_\d+', c)]
    user_columns = [c for c in columns if re.match(r'u_\d+', c)]
    rating_column = 'rating'
    return df[movie_columns].to_numpy(), df[user_columns].to_numpy(), df[rating_column].to_numpy()


def cos_to_rating(x):
    add, mul = tf.math.add, tf.math.multiply
    return mul(add(x, 1), 2.5)


def get_m1(name='m1'):
    m_input = ks.Input(200, name='movie')
    md1 = ks.layers.Dense(10000, activation='relu', name='md1')
    md2 = ks.layers.Dense(5000, activation='relu', name='md2')
    md3 = ks.layers.Dense(10000, activation='relu', name='md3')
    md4 = ks.layers.Dense(50, activation='linear', name='md4')

    u_input = ks.Input(200, name='user')
    ud1 = ks.layers.Dense(10000, activation='relu', name='ud1')
    ud2 = ks.layers.Dense(5000, activation='relu', name='ud2')
    ud3 = ks.layers.Dense(10000, activation='relu', name='ud3')
    ud4 = ks.layers.Dense(50, activation='linear', name='ud4')

    dot = ks.layers.Dot(axes=1, normalize=True)

    m_br = md4(md3(md2(md1(m_input))))
    u_br = ud4(ud3(ud2(ud1(u_input))))
    output = cos_to_rating(dot([m_br, u_br]))

    return ks.Model(inputs=[m_input, u_input], outputs=output, name=name)


def export_model_architectures(models):
    for model in models:
        ks.utils.plot_model(model, f'{model_architecture_path}/emb_cb_filtering_{model.name}.png', show_shapes=True,
                            show_layer_names=True, show_layer_activations=True)


def main():
    np.random.seed(0)
    tf.random.set_seed(0)
    train_df, dev_df, test_df = load_samples(200, 5_000, 2_500, 2_500)

    train_movies, train_users, train_ratings = prepare_data(train_df)
    dev_movies, dev_users, dev_ratings = prepare_data(dev_df)

    models = [get_m1()]
    histories = []
    predictions = []

    export_model_architectures(models)

    for model in models:
        model.compile(
            optimizer=tf.optimizers.Adam(0.0000001),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanSquaredError('mse')]
        )
        history = model.fit(
            x={'movie': train_movies, 'user': train_users},
            y=train_ratings,
            epochs=30,
            validation_data=[{'movie': dev_movies, 'user': dev_users}, dev_ratings]
        )
        prediction = model.predict({'movie': dev_movies, 'user': dev_users})
        histories.append(history)
        predictions.append(prediction
                           )
    compare_history(histories, ['mse', 'val_mse'])
    compare_distributions(dev_ratings, [(m.name, p) for m, p in zip(models, predictions)], bins=50, rwidth=1)


if __name__ == '__main__':
    main()
