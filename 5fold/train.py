import os
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
from unet_model import create_unet_model
from loader import get_train_val_loaders
from radam import RAdam
import settings

train_on_gpu = True


def train(args):
    ckp = None
    if os.path.exists(args.log_dir + '/checkpoints/best.pth'):
        ckp = args.log_dir + '/checkpoints/best.pth'
    model = create_model(args.encoder_type, ckp=ckp)
    loaders = get_train_val_loaders(args.encoder_type, batch_size=args.batch_size, ifold=args.ifold)

    # model, criterion, optimizer
    if args.encoder_type.startswith('myunet'):
        optimizer = RAdam(model.parameters(), lr=args.lr)
    else:
        optimizer = RAdam([
            {'params': model.decoder.parameters(), 'lr': args.lr}, 
            {'params': model.encoder.parameters(), 'lr': args.lr / 10.},  
        ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    callbacks = [
        DiceCallback(), 
        EarlyStoppingCallback(patience=20, min_delta=0.001), 
    ]
    #if os.path.exists(args.log_dir + '/checkpoints/best_full.pth'):
    #    callbacks.append(CheckpointCallback(resume=args.log_dir + '/checkpoints/best_full.pth'))

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=args.log_dir,
        num_epochs=args.num_epochs,
        verbose=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--encoder_type', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--iter_val', default=400, type=int, help='start epoch')
    parser.add_argument('--num_epochs', default=60, type=int, help='epoch')
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
