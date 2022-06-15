from tqdm.notebook import tqdm
import torch


def train(model, num_epochs, train_loader, device, lr_scheduler, optimizer, loss, num_save=3):
    loss_epochs = []
    loss_iter = []
    loss_iter_epochs = []
    num_model = 1

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

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % num_save == 0:
            name_model = rf'..\..\models\fasterrcnn_model_{num_model}.pth'
            torch.save(model.state_dict(), name_model)
            print(f'Model {name_model} has been saved')
            print('\n')
            num_model += 1
