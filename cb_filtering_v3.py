import tensorflow as tf
from tensorflow import keras as ks

from data import read_train_data, read_dev_data
from utils import compare_history, compare_distributions

model_architecture_path = 'report/'


def build_model_3():
    regularization = 0.01

    movie_input = ks.Input(21, name='movie')
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(movie_input)
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(x)
    movie_node = ks.layers.Dense(50, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(x)

    user_input = ks.Input(20, name='user')
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(user_input)
    x = ks.layers.Dense(500, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(x)
    user_node = ks.layers.Dense(50, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(x)

    dot_node = ks.layers.Dot(1, name='rating')

    output = dot_node([movie_node, user_node])

    return ks.Model(inputs=[movie_input, user_input], outputs=[output], name='m3')


def build_model_6():
    regularization = 0.1

    movie_input = ks.Input(21, name='movie')
    x = ks.layers.Dense(200, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(movie_input)
    x = ks.layers.Dense(200, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(x)
    movie_node = ks.layers.Dense(30, activation='linear', kernel_regularizer=ks.regularizers.L2(regularization))(x)

    user_input = ks.Input(20, name='user')
    x = ks.layers.Dense(200, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(user_input)
    x = ks.layers.Dense(200, activation='relu', kernel_regularizer=ks.regularizers.L2(regularization))(x)
    user_node = ks.layers.Dense(30, activation='linear', kernel_regularizer=ks.regularizers.L2(regularization))(x)

    dot_node = ks.layers.Dot(1, name='rating', normalize=True)

    x = dot_node([movie_node, user_node])
    x = tf.math.add(x, 1, 'shift')
    output = tf.math.multiply(x, 2.5, 'scale')

    return ks.Model(inputs=[movie_input, user_input], outputs=[output], name='m6')


def export_model_architectures(models):
    for model in models:
        ks.utils.plot_model(model, f'{model_architecture_path}/cb_filtering_v3_{model.name}.png', show_shapes=True,
                            show_layer_names=True, show_layer_activations=True)


def main():
    train_movie_df, train_user_df, train_rating_df, train_meta_df = read_train_data(load_meta=False)
    dev_movie_df, dev_user_df, dev_rating_df, dev_meta_df = read_dev_data(load_meta=False)

    train_set_size = 50000
    dev_set_size = 25000
    train_movie_df = train_movie_df[:train_set_size]
    train_user_df = train_user_df[:train_set_size]
    train_rating_df = train_rating_df[:train_set_size]
    dev_movie_df = dev_movie_df[:dev_set_size]
    dev_user_df = dev_user_df[:dev_set_size]
    dev_rating_df = dev_rating_df[:dev_set_size]

    models = [build_model_3(), build_model_6()]
    histories = []
    predictions = []

    export_model_architectures([m for m in models if m.name in []])

    for model in models:
        print(f'Fitting: {model.name}'.center(98, '-'))
        model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=0.0001),
            loss=ks.losses.MeanSquaredError(),
            metrics=[ks.metrics.MeanSquaredError(name='mse')]
        )
        history = model.fit(
            {'movie': train_movie_df, 'user': train_user_df},
            train_rating_df,
            epochs=30,
            validation_data=({'movie': dev_movie_df, 'user': dev_user_df}, dev_rating_df)
        )
        prediction = model.predict({'movie': dev_movie_df, 'user': dev_user_df})

        histories.append(history)
        predictions.append([model.name, prediction])
        print(f'End of fitting: {model.name}'.center(98, '-'))

    compare_history(histories, ['mse', 'val_mse'])
    compare_distributions(dev_rating_df, predictions)


if __name__ == '__main__':
    main()
