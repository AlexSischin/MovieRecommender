from tensorflow import keras as ks

from data import read_train_data, read_dev_data
from utils import compare_history

model_architecture_path = 'report/'


def build_model_1():
    movie_input = ks.Input(21, name='movie')
    x = ks.layers.Dense(200, activation='relu')(movie_input)
    x = ks.layers.Dense(200, activation='relu')(x)
    movie_node = ks.layers.Dense(30, activation='relu')(x)

    user_input = ks.Input(20, name='user')
    x = ks.layers.Dense(200, activation='relu')(user_input)
    x = ks.layers.Dense(200, activation='relu')(x)
    user_node = ks.layers.Dense(30, activation='relu')(x)

    dot_node = ks.layers.Dot(1, name='rating')

    output = dot_node([movie_node, user_node])

    return ks.Model(inputs=[movie_input, user_input], outputs=[output], name='m1')


def build_model_2():
    movie_input = ks.Input(21, name='movie')
    x = ks.layers.Dense(500, activation='relu')(movie_input)
    x = ks.layers.Dense(200, activation='relu')(x)
    movie_node = ks.layers.Dense(50, activation='relu')(x)

    user_input = ks.Input(20, name='user')
    x = ks.layers.Dense(500, activation='relu')(user_input)
    x = ks.layers.Dense(200, activation='relu')(x)
    user_node = ks.layers.Dense(50, activation='relu')(x)

    dot_node = ks.layers.Dot(1, name='rating')

    output = dot_node([movie_node, user_node])

    return ks.Model(inputs=[movie_input, user_input], outputs=[output], name='m2')


def build_model_3():
    movie_input = ks.Input(21, name='movie')
    x = ks.layers.Dense(500, activation='relu')(movie_input)
    x = ks.layers.Dense(500, activation='relu')(x)
    movie_node = ks.layers.Dense(50, activation='relu')(x)

    user_input = ks.Input(20, name='user')
    x = ks.layers.Dense(500, activation='relu')(user_input)
    x = ks.layers.Dense(500, activation='relu')(x)
    user_node = ks.layers.Dense(50, activation='relu')(x)

    dot_node = ks.layers.Dot(1, name='rating')

    output = dot_node([movie_node, user_node])

    return ks.Model(inputs=[movie_input, user_input], outputs=[output], name='m3')


def build_model_4():
    movie_input = ks.Input(21, name='movie')
    x = ks.layers.Dense(1000, activation='relu')(movie_input)
    x = ks.layers.Dense(1000, activation='relu')(x)
    x = ks.layers.Dense(500, activation='relu')(x)
    movie_node = ks.layers.Dense(50, activation='relu')(x)

    user_input = ks.Input(20, name='user')
    x = ks.layers.Dense(1000, activation='relu')(user_input)
    x = ks.layers.Dense(1000, activation='relu')(x)
    x = ks.layers.Dense(500, activation='relu')(x)
    user_node = ks.layers.Dense(50, activation='relu')(x)

    dot_node = ks.layers.Dot(1, name='rating')

    output = dot_node([movie_node, user_node])

    return ks.Model(inputs=[movie_input, user_input], outputs=[output], name='m4')


def export_model_architectures(models):
    for model in models:
        ks.utils.plot_model(model, f'{model_architecture_path}/cb_filtering_{model.name}.png', show_shapes=True,
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

    models = [build_model_1(), build_model_2(), build_model_3(), build_model_4()]
    histories = []

    export_model_architectures(models)

    for model in models:
        print(f'Fitting: {model.name}'.center(98, '-'))
        model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=0.001),
            loss=ks.losses.MeanSquaredError(),
            metrics=[ks.metrics.MeanSquaredError(name='mse')]
        )
        history = model.fit(
            {'movie': train_movie_df, 'user': train_user_df},
            train_rating_df,
            epochs=30,
            validation_data=({'movie': dev_movie_df, 'user': dev_user_df}, dev_rating_df)
        )
        histories.append(history)
        print(f'End of fitting: {model.name}'.center(98, '-'))

    compare_history(histories, ['mse', 'val_mse'])


if __name__ == '__main__':
    main()
