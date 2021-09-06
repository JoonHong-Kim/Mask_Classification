import os
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import cv2
import torchvision
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

"""
class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD
"""


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = torchvision.transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


class C_TrainDataset(Dataset):
    num_classes = 3 * 2 * 3
    mean = (0.548, 0.504, 0.479)
    std = (0.237, 0.247, 0.246)

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        info_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

        self.image_dir = os.path.join(data_dir, 'cropped')
        self.data, self.targets = self._load_data()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img_name = self.data[index]
        img = Image.open(img_name)
        img = np.array(img)
        if img.shape[2] == 4:
            img = Image.fromarray(np.delete(img, -1, axis=-1))
        image = self.transform(image=img)['image']
        return image, self.targets[index]

    def set_transform(self, transforms):
        self.transform = transforms

    # 000001_female_Asian_45incorrect_mask.jpg
    def _load_data(self):
        data = []
        target = []

        for _img_name in os.listdir(self.image_dir):
            if _img_name[0] == '.':
                continue
            label = 0
            img_name = _img_name.split('_')
            sex = img_name[1]

            age = int(img_name[3][:2])

            mask = img_name[3][2:]
            if sex == "female":
                label += 3
            if 30 <= age < 59:
                label += 1
            elif age >= 59:
                label += 2
            if "incorrect" in mask:
                label += 6
            if "normal" in mask:
                label += 12
            target.append(label)
            data.append(os.path.join(self.image_dir, _img_name))
        return data, target


class TrainDataset(Dataset):
    """
    작성자: 김준홍_T2059
    Train,Validation Dataset
    """
    num_classes = 3 * 2 * 3
    mean = (0.548, 0.504, 0.479)
    std = (0.237, 0.247, 0.246)

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        info_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

        image_dir = os.path.join(data_dir, 'new_imgs')
        img_ids = [info_train['id'][i] + '_' + info_train['gender'][i] + '_Asian_' + str(info_train['age'][i]) for i in
                   info_train.index]
        self.img_paths = [os.path.join(image_dir, i_p) for i_p in img_ids]
        self.data, self.targets = self._load_data()

    def __getitem__(self, index):
        img_name = self.data[index]
        img = Image.open(img_name)
        img = np.array(img)
        if img.shape[2] == 4:
            img = Image.fromarray(np.delete(img, -1, axis=-1))
        image = self.transform(image=img)['image']
        return image, self.targets[index]

    def set_transform(self, transforms):
        self.transform = transforms

    def __len__(self):
        return len(self.targets)

    def _load_data(self):
        data = []
        target = []
        for img_path in self.img_paths:
            # img path에 있는 file들 불러옴
            dir_name, _, files = next(iter(os.walk(img_path)))
            # [id,sex,race,age]
            info = dir_name.split('/')[-1].split('_')
            # labeling
            label = 0
            if info[1] == 'female':
                label += 3
            if 30 <= int(info[-1]) < 59:
                label += 1
            elif int(info[-1]) >= 59:
                label += 2
            for file_name in files:
                if file_name[0] == '.':
                    continue
                if 'incorrect' in file_name:
                    target.append(label + 6)
                elif 'mask' in file_name:
                    target.append(label)
                else:
                    target.append(label + 12)
                data.append(os.path.join(dir_name, file_name))
        return data, target


class CustomAugmentation:
    """
    제작자: 김준홍_T2059
    CustomAugmentation -> special mission 참고
    """

    def __init__(self, need, resize, mean, std, **args):
        if 'train' in need:
            self.transform = Compose([
                Resize(resize[0], resize[1], p=1.0),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                GaussNoise(p=0.5),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        elif 'val' in need:
            self.transform = Compose([
                Resize(resize[0], resize[1], p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
        else:
            raise ValueError("You have to train or val in need parameters")

    def __call__(self, image):
        return self.transform(image=np.array(image))
