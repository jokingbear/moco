import numpy as np
import cv2

from plasma.training.data import Dataset, augmentations as augs
from albumentations import HorizontalFlip, OneOf, RandomBrightnessContrast, RandomGamma, ShiftScaleRotate, Compose, \
    CoarseDropout

aug = Compose([
    CoarseDropout(min_holes=0, max_holes=3, max_height=32, max_width=32, p=0.8),
    augs.MinEdgeCrop(always_apply=True),
    HorizontalFlip(),
    OneOf([
        RandomGamma(),
        RandomBrightnessContrast(),
    ], p=0.8),
    ShiftScaleRotate(shift_limit=0.1, rotate_limit=35, scale_limit=0.2, p=0.8, border_mode=cv2.BORDER_CONSTANT),
])


class Data(Dataset):

    def __init__(self, df, image_path):
        super().__init__()

        self.df = df.copy().reset_index(drop=True)
        self.image_path = image_path

    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]
        path = row[self.image_path]

        img = np.load(path)

        img1 = aug(image=img)["image"]
        img2 = aug(image=img)["image"]

        return img1[np.newaxis], img2[np.newaxis]
