import torch
from src.visualization import visualize


def valid_model_predict(model, valid_loader, device):
    model.eval()
    iteration = 0
    for imgs, annotations in valid_loader:
        iteration += 1
        imgs = [img.to(device) for img in imgs]
        annotations = [{key:value.to(device) for key, value in annotation.items()} for annotation in annotations]

        with torch.no_grad():
            prediction = model(imgs, annotations)
            losses = sum(loss for loss in prediction.values())

    print(losses/iteration)


def get_rezult(model,valid_loader, device):
    valid_imgs, annotations_valid = next(iter(valid_loader))
    valid_imgs = [valid_img.to(device) for valid_img in valid_imgs]
    annotations_valid = [{key: value.to(device) for key, value in annotation_valid.items()} for annotation_valid in annotations_valid]

    model.eval()
    test_device = torch.device('cpu')
    model = model.to(device)
    outputs_imgs = model(valid_imgs)
    outputs = [{key:value for key, value in out.items()} for out in outputs_imgs]
    numb_img = 0
    visualize.vizualize_box(valid_imgs[numb_img], outputs[numb_img])
    torch.save(model.state_dict(), r'..\..\models\fasterrcnn_global_whea_detect.pth')