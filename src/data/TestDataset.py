import numpy as np
from torch.utils.data import Dataset
import cv2


class DatasetWheatTest(Dataset):

    def __init__(self, data_frame, dir_imgs, transforms=None):
        super(Dataset).__init__()

        self.imgs_id = data_frame['image_id'].unique()
        self.df = data_frame
        self.dir_imgs = dir_imgs
        self.transforms = transforms

    def __getitem__(self, idx):

        image_id = self.imgs_id[idx]
        records_img = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f"{self.dir_imgs}/{image_id}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms is not None:
            sample = {
                'image': image,
            }

            sample = self.transforms(**sample)

            image = sample['image']

        return image

    def __len__(self):
        return self.imgs_id.shape[0]
