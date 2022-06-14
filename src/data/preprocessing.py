import numpy as np
import re
import pandas as pd
from typing import Text
import yaml
import argparse


def main_preprocessing(config_path: Text) -> None:
    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)
    df = pd.read_csv(config['data_split']['trainset_path'])
    df = convert_df(df)
    valid_df, train_df = data_split(df)
    valid_df.to_csv(config['data_split']['validset_path'])
    train_df.to_csv(config['data_split']['new_trainset_path'])
    print("complete!")


def convert_df(df) -> pd.DataFrame:
    df['x'] = -1
    df['y'] = -1
    df['w'] = -1
    df['h'] = -1

    df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda obj: expend_box(obj)))
    df['x'] = df['x'].astype(np.float64)
    df['y'] = df['y'].astype(np.float64)
    df['w'] = df['w'].astype(np.float64)
    df['h'] = df['h'].astype(np.float64)

    df.drop(columns=['bbox'], inplace=True)

    return df


def data_split(df):
    imgs_id = df['image_id'].unique()
    print(len(imgs_id))
    perct = 0.2
    ids = int(len(imgs_id) * (1 - perct))
    print(ids)
    train_ids = imgs_id[:ids]
    valid_ids = imgs_id[ids:]
    # print(train_ids, valid_ids)
    valid_df = df[df['image_id'].isin(valid_ids)]
    # print(len(valid_df.image_id.unique()))
    train_df = df[df['image_id'].isin(train_ids)]
    print(len(train_df.image_id.unique()), len(valid_df.image_id.unique()))
    return valid_df, train_df


def expend_box(obj) -> np.array:
    temple = np.array(re.findall("([0-9]+[.]?[0-9]*)", obj))
    if len(temple) == 0:
        temple = [-1, -1, -1, -1]
    return temple


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()

    main_preprocessing(config_path=args.config)