import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import albumentations as albu
from albumentations import torch as AT

import settings

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightnessContrast,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Resize, RandomSizedCrop,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, VerticalFlip,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)

class Rotate90(RandomRotate90):
    def apply(self, img, factor=1, **params):
        return np.ascontiguousarray(np.rot90(img, factor))

def img_augment(p=.8):
    return Compose([
        RandomSizedCrop((280, 345), 350, 525, p=0.9, w2h_ratio=1.5),
        HorizontalFlip(.5),
        VerticalFlip(.5),
        #RandomRotate90(p=0.2),
        OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
        #
        ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=20, p=.75 ),
        Blur(blur_limit=3, p=.33),
        OpticalDistortion(p=.33),
        GridDistortion(p=.33),
        #HueSaturationValue(p=.33)
    ], p=p)

def weak_augment(p=.8):
    return Compose([
        Resize(320, 640, p=1.),
        RandomSizedCrop((200, 250), 256, 256, p=0.8),
        RandomRotate90(p=0.05),
        OneOf([
                #CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
        #
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=.75 ),
        Blur(blur_limit=3, p=.33),
        OpticalDistortion(p=.33),
        #GridDistortion(p=.33),
        #HueSaturationValue(p=.33)
    ], p=p)
class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train'):
        self.df = df
        self.datatype = datatype
        if datatype != 'test':
            self.data_folder = f"{settings.DATA_DIR}/train"
        else:
            self.data_folder = f"{settings.DATA_DIR}/test"

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['Image']
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (525, 350))

        if self.datatype == 'train':
            aug = img_augment(p=0.8)
            img = aug(image=img)['image']
        
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        target = torch.tensor(self.df.iloc[idx][class_names].values.astype(np.float32))
        
        return img, target

    def __len__(self):
        return len(self.df)

def prepare_df():
    train_df = pd.read_csv(f'{settings.DATA_DIR}/train.csv')
    train_df = train_df[~train_df['EncodedPixels'].isnull()]
    train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
    train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
    classes = train_df['Class'].unique()
    train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
    #print(train_df.head())
    #print(train_df.iloc[0][class_names].values.astype(np.float32))
    #train_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'))['ids'].values.tolist()
    #valid_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'val_ids.csv'))['ids'].values.tolist()
    return train_df


def get_train_val_loaders(batch_size=16, val_percent=0.1):
    df = prepare_df().sort_values(by='Image')
    df = shuffle(df, random_state=1234)
    #print(train.head())
    #num_workers = 16
    split_index = int(len(df) * (1-val_percent))
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]
    print('train:', len(train_df), 'val:', len(val_df))
    print(val_df.head())

    train_dataset = CloudDataset(df=train_df, datatype='train')
    valid_dataset = CloudDataset(df=val_df, datatype='valid')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    return loaders

def get_test_loader(batch_size=16):
    sub = pd.read_csv(os.path.join(settings.DATA_DIR, 'sample_submission.csv'))
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    return test_loader

def test_prepare_df():
    train, train_ids, val_ids1 = prepare_df()
    train, train_ids, val_ids2 = prepare_df()
    
    print(len(set(val_ids1)), len(set(val_ids2)), len(set(val_ids1) & set(val_ids2)), len(set(val_ids1) - set(val_ids2)))
    print(sorted(val_ids1[:50]))
    print(sorted(val_ids1[-50:]))

def test_ds():
    train_loader = get_train_val_loaders()['train']
    for batch in train_loader.dataset:
        print(batch)
        break

def test_train_loader():
    train_loader = get_train_val_loaders(batch_size=4)['train']

    for x, t in train_loader:
        print(x, x.size(), t, t.size())
        break

def test_test_loader():
    loader = get_test_loader()
    for x in loader:
        print(x[0].size(), x[1].size())
        break

if __name__ == '__main__':
    #test_ds()
    test_train_loader()
    #test_prepare_df()
    #test_test_loader()

