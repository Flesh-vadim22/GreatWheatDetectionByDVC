import pandas as pd
import argparse
import yaml
from typing import Text
import sys
import torch
sys.path.append('src/')
from data.TrainDataset import TrainDataset
from features.transforms import transform_train, transform_test
from model.set_model import get_model
from utils.data_settings import collate_fn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.notebook import tqdm


def train_model(config_path: Text) -> None:

    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)

    device = config['base']['device']['GPU'] if torch.cuda.is_available() else config['base']['device']['CPU']
    num_classes = config['base']['num_classes']
    model = get_model(num_classes)
    model.to(device)

    train_df = pd.read_csv(config['data_split']['new_trainset_path'])

    train_dataset = TrainDataset(train_df, config['data_split']['train_path_data'], transforms=transform_train())

    train_batch_size = config['data_split']['train_batch_size']

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_scheduler = None

    num_epochs = config['train']['num_epochs']
    num_model = config['train']['num_model']

    loss_epochs = []
    loss_iter = []
    loss_iter_epochs = []

    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        iteretions = 0
        for imgs, annotations in tqdm(train_loader):
            iteretions += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{key: value.to(device) for key, value in annotation.items()} for annotation in annotations]

            prediction = model(imgs, annotations)

            losses = sum(loss for loss in prediction.values())
            epoch_loss += losses.item()

            loss_iter.append(losses.item()/iteretions)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if iteretions % 40 == 0:
                print(f'For iteretion {iteretions} epoch {epoch + 1} loss is {losses.item()}')

        print(f'For epoch {epoch + 1} loss is {epoch_loss/iteretions}')
        loss_epochs.append(epoch_loss/iteretions)
        loss_iter_epochs.append(loss_iter)

        if scheduler is not None:
            scheduler.step()

        if epoch % 3 == 0:
            name_model = f"{config['train']['model_path']}fasterrcnn_model_{num_model}.pth"
            torch.save(model.state_dict(), name_model)
            print(f'Model {name_model} has been saved')
            print('\n')
            num_model += 1

    num_model += 1
    name_model = f"{config['train']['model_path']}fasterrcnn_model_{num_model}.pth"
    torch.save(model.state_dict(), name_model)
    print(f'Model {name_model} has been saved')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    ars = arg_parser.parse_args()
    train_model(config_path=ars.config)



