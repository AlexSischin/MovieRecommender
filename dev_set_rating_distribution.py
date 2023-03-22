from data import read_dev_data
from utils import compare_distributions

if __name__ == '__main__':
    dev_movie_df, dev_user_df, dev_rating_df, dev_meta_df = read_dev_data(load_meta=False)
    compare_distributions(dev_rating_df, [], bins=30)
