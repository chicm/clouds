import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
import segmentation_models_pytorch as smp

from models import create_model
from loader import get_train_val_loaders
from radam import RAdam
import settings

train_on_gpu = True


def train(args):
    num_epochs = 100
    logdir = "./logs/segmentation"

    model = create_model()
    loaders = get_train_val_loaders(batch_size=args.batch_size)

    # model, criterion, optimizer
    optimizer = RAdam([
        {'params': model.decoder.parameters(), 'lr': 1e-2}, 
        {'params': model.encoder.parameters(), 'lr': 1e-3},  
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[
            DiceCallback(), 
            EarlyStoppingCallback(patience=25, min_delta=0.001), 
            #CheckpointCallback(resume='./logs/segmentation/checkpoints/best_full.pth')
            ],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='learning rate')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--iter_val', default=400, type=int, help='start epoch')
    parser.add_argument('--num_epochs', default=1, type=int, help='epoch')
    parser.add_argument('--optim_name', default='RAdam', choices=['SGD', 'RAdam', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=8, type=int, help='lr scheduler patience')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--ifold', default=0, type=int, help='lr scheduler patience')
    
    args = parser.parse_args()
    print(args)
    train(args)
