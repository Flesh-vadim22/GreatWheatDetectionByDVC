import pandas as pd
from src.data.TestDataset import DatasetWheatTest
from torch.utils.data import DataLoader
import torch
import numpy as np
from src.features.transforms import transform_test
from src.utils.data_settings import collate_fn


def inference(model, device):
    test_df = pd.read_csv(r'..\..\data\processed\psample_submission.csv')
    DIR_TEST = r'..\..\data\raw\valid'
    test_dataset = DatasetWheatTest(test_df, DIR_TEST,transforms=transform_test())
    test_loader = DataLoader(test_dataset,batch_size=4, shuffle=False, num_workers=4, drop_last=False,collate_fn=collate_fn)

    threshold_detect = 0.5
    result = []
    for imgs, annotations in test_loader:
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            prediction = model(imgs)

        for i, image in enumerate(imgs):
            boxes = prediction[i]['boxes'].data.cpu().numpy()
            scores = prediction[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores>=threshold_detect].as_type(np.float32)
            scores = scores[scores>=threshold_detect]
            img_id = image[i]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            str_result = {
                'image_id': img_id ,
                'PredictionString':format_submit_string(boxes, scores),
            }

            result.append(str_result)


def format_submit_string(boxes, scores):
    submit_string = []
    for i in zip(boxes, scores):
        submit_string.append("{0:.4f} {1} {2} {3} {4}".format(i[0], i[1][0], i[1][1], i[1][2], i[1][3]))

    return "".join(submit_string)