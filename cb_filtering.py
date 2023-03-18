from data import read_train_data

if __name__ == '__main__':
    train_movie_df, train_user_df, train_rating_df, train_meta_df = read_train_data(load_meta=False)

