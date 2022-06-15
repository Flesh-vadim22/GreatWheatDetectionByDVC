import pandas as pd
import argparse
import yaml
from typing import Text
import sys
import torch
sys.path.append('src/')
from data.Testdataset import Testdataset
from features.transforms import transform_test
from torch.utils.data import DataLoader


def evaluate(config_path: Text):
    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)
    model = torch.load(f"{config['train']['model_path']}fasterrcnn_model_1.pth")
    valid_df = pd.read_csv(config['data_split']['validset_path'])
    valid_dataset = Testdataset(valid_df, config['data_split']['train_path_data'], transforms=transform_test())
    train_batch_size = config['data_split']['valid_batch_size']
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=True)

    device = config['base']['device']['GPU'] if torch.cuda.is_available() else config['base']['device']['CPU']
    model.to(device)
    model.eval()
    iteration = 0
    for imgs, annotations in valid_loader:
        iteration += 1
        imgs = [img.to(device) for img in imgs]
        annotations = [{key: value.to(device) for key, value in annotation.items()} for annotation in annotations]

        with torch.no_grad():
            prediction = model(imgs, annotations)
        losses = sum(loss for loss in prediction.values())

    print(losses / iteration)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    ars = arg_parser.parse_args()
    evaluate(config_path=ars.config)
