import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class TrainDataset(Dataset):

    def __init__(self, data_frame, dir_imgs, transforms=None):
        super(Dataset).__init__()

        self.imgs_id = data_frame['image_id'].unique()
        self.df = data_frame
        self.dir_imgs = dir_imgs
        self.transforms = transforms

    def __getitem__(self, idx):

        image_id = self.imgs_id[idx]
        records_img = self.df[self.df['image_id'] == image_id]
        image = cv2.imread(f"{self.dir_imgs}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image/=255.0
        boxes = records_img[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.ones((records_img.shape[0],), dtype=torch.int64)

        iscrowd = torch.zeros((records_img.shape[0],),dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels,
            }

            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)

        return image, target

    def __len__(self):
        return self.imgs_id.shape[0]
