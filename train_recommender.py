import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as ks

from data import read_train_data, read_dev_data, read_test_data
from utils import compare_history, compare_distributions

model_architecture_path = 'report/'


def create_model(name='m7'):
    out_reg = 0.01
    hidden_reg = 0.1

    movie_input = ks.Input(21, name='movie')
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(out_reg))(movie_input)
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(hidden_reg))(x)
    movie_node = ks.layers.Dense(50, activation='linear', kernel_regularizer=ks.regularizers.L2(out_reg))(x)

    user_input = ks.Input(20, name='user')
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(out_reg))(user_input)
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(hidden_reg))(x)
    user_node = ks.layers.Dense(50, activation='linear', kernel_regularizer=ks.regularizers.L2(out_reg))(x)

    dot_node = ks.layers.Dot(1, name='rating', normalize=True)

    x = dot_node([movie_node, user_node])
    x = tf.math.add(x, 1, 'shift')
    output = tf.math.multiply(x, 2.5, 'scale')

    return ks.Model(inputs=[movie_input, user_input], outputs=[output], name=name)


def main():
    train_movie_df, train_user_df, train_rating_df, train_meta_df = read_train_data(load_meta=False)
    dev_movie_df, dev_user_df, dev_rating_df, dev_meta_df = read_dev_data(load_meta=False)
    test_movie_df, test_user_df, test_rating_df, test_meta_df = read_test_data(load_meta=False)

    train_set_size = 1_000_000
    dev_set_size = 50_000
    test_set_size = 1_000_000
    train_movies = train_movie_df.to_numpy()[:train_set_size]
    train_users = train_user_df.to_numpy()[:train_set_size]
    train_ratings = train_rating_df.to_numpy()[:train_set_size]
    dev_movies = dev_movie_df.to_numpy()[:dev_set_size]
    dev_users = dev_user_df.to_numpy()[:dev_set_size]
    dev_ratings = dev_rating_df.to_numpy()[:dev_set_size]
    test_movies = test_movie_df.to_numpy()[:test_set_size]
    test_users = test_user_df.to_numpy()[:test_set_size]
    test_ratings = test_rating_df.to_numpy()[:test_set_size]

    movie_scaler = StandardScaler().fit(train_movies)
    train_movies = movie_scaler.transform(train_movies)
    dev_movies = movie_scaler.transform(dev_movies)
    test_movies = movie_scaler.transform(test_movies)

    user_scaler = StandardScaler().fit(train_users)
    train_users = user_scaler.transform(train_users)
    dev_users = user_scaler.transform(dev_users)
    test_users = user_scaler.transform(test_users)

    model = create_model()
    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=0.001),
        loss=ks.losses.MeanSquaredError(),
        metrics=[ks.metrics.MeanSquaredError(name='mse')]
    )
    early_stop = ks.callbacks.EarlyStopping(
        min_delta=10e-4,
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    history = model.fit(
        {'movie': train_movies, 'user': train_users},
        train_ratings,
        epochs=30,
        callbacks=[early_stop],
        validation_data=({'movie': dev_movies, 'user': dev_users}, dev_ratings)
    )
    test_results = model.evaluate({'movie': test_movies, 'user': test_users}, test_ratings, return_dict=True)
    prediction = model.predict({'movie': test_movies, 'user': test_users})

    print(f'Test set MSE: {test_results["mse"]:.3f}')
    ks.utils.plot_model(model, f'{model_architecture_path}/cb_filtering_v5_{model.name}.png', show_shapes=True,
                        show_layer_names=True, show_layer_activations=True)
    compare_history([history], ['mse', 'val_mse'])
    compare_distributions(test_ratings, [(model.name, prediction)])

    model.save('model/m7')


if __name__ == '__main__':
    main()
