import numpy as np
import re
import pandas as pd


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


def data_split(df) -> tuple:
    imgs_id = df['image_id'].unique()
    perct = 0.2
    ids = int(len(df) * (1 - perct))
    # print(ids)
    train_ids = imgs_id[:ids]
    valid_ids = imgs_id[ids:]
    # print(len(imgs_id), len(valid_ids), len(train_ids))
    valid_df = df[df['image_id'].isin(valid_ids)]
    train_df = df[df['image_id'].isin(train_ids)]
    print(len(train_df.image_id.unique()), len(valid_df.image_id.unique()))
    return valid_df, train_df


def expend_box(obj) -> np.array:
    temple = np.array(re.findall("([0-9]+[.]?[0-9]*)", obj))
    if len(temple) == 0:
        temple = [-1, -1, -1, -1]
    return temple
