from data import load_emb_data


def main():
    train_df, dev_df, test_df = next(zip(*load_emb_data(25, 50_000, 25_000, 25_000)))
    pass


if __name__ == '__main__':
    main()
