import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def transform_train():
    return A.Compose([
        A.Flip(p=1.0),
        ToTensorV2(p=1.0)], bbox_params={'format':'pascal_voc', 'label_fields': ['labels']})


def transform_test():
    return A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

