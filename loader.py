import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import torch as AT

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.contrib.models.segmentation import Unet
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

import segmentation_models_pytorch as smp
from utils import get_training_augmentation, get_validation_augmentation, get_preprocessing, make_mask
import settings

class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]),
                preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{settings.DATA_DIR}/train"
        else:
            self.data_folder = f"{settings.DATA_DIR}/test"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

def prepare_df():
    train = pd.read_csv(f'{settings.DATA_DIR}/train.csv')
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    #train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    #test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
    #sub = pd.read_csv(f'{path}/sample_submission.csv')
    train_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_ids.csv'))['ids'].values.tolist()
    valid_ids = pd.read_csv(os.path.join(settings.DATA_DIR, 'val_ids.csv'))['ids'].values.tolist()
    return train, train_ids, valid_ids


def get_train_val_loaders(encoder_type, batch_size=16):
    if encoder_type.startswith('myunet'):
        encoder_type = 'resnet50'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_type, 'imagenet')
    train, train_ids, valid_ids = prepare_df()
    num_workers = 24
    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader.num = len(train_ids)
    valid_loader.num = len(valid_ids)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    return loaders

def get_test_loader(encoder_type, batch_size=16):
    if encoder_type.startswith('myunet'):
        encoder_type = 'resnet50'
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_type, 'imagenet')

    sub = pd.read_csv(os.path.join(settings.DATA_DIR, 'sample_submission.csv'))
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=24)

    return test_loader

def test_prepare_df():
    train, train_ids, val_ids1 = prepare_df()
    train, train_ids, val_ids2 = prepare_df()
    
    print(len(set(val_ids1)), len(set(val_ids2)), len(set(val_ids1) & set(val_ids2)), len(set(val_ids1) - set(val_ids2)))
    print(sorted(val_ids1[:50]))
    print(sorted(val_ids1[-50:]))

def test_ds():
    train_loader = get_train_val_loaders('densenet201')['train']
    for batch in train_loader.dataset:
        print(batch)
        break

def test_train_loader():
    train_loader = get_train_val_loaders('densenet201')['train']

    for x in train_loader:
        print(x)
        break
    print(dir(train_loader))
    print(train_loader.dataset.img_ids[:50])

def test_test_loader():
    loader = get_test_loader('densenet201')
    for x in loader:
        print(x[0].size(), x[1].size())
        break

if __name__ == '__main__':
    #test_ds()
    #test_train_loader()
    #test_prepare_df()
    test_test_loader()

