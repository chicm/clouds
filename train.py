import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl.core import MetricCallback
from catalyst.dl.utils.criterion.dice import dice
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import BCEDiceLoss

from models import create_model
from loader import get_train_val_loaders
from radam import RAdam
import time
import settings

train_on_gpu = True

class MixedLoss(nn.Module):
    __name__ = 'mixed_loss'

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_dice = BCEDiceLoss()

    def forward(self, y_pr, y_gt):
        mask_out, cls_out = y_pr
        mask_target, cls_target = y_gt

        dice_loss = self.bce_dice(mask_out, mask_target)
        cls_loss = self.bce(cls_out, cls_target)

        return dice_loss, cls_loss

#c = MixedLoss()
c = BCEDiceLoss()

def criterion(y_pred, y_true):
    return c(y_pred, y_true)


def train(args):
    model, model_file = create_model(args.encoder_type, work_dir=args.work_dir, ckp=args.ckp)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.cuda()

    loaders = get_train_val_loaders(args.encoder_type, batch_size=args.batch_size)

    #optimizer = RAdam([
    #    {'params': model.decoder.parameters(), 'lr': args.lr}, 
    #    {'params': model.encoder.parameters(), 'lr': args.lr / 10.},  
    #])
    if args.optim_name == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=args.lr)
    elif args.optim_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)


    best_metrics = 0.
    best_key = 'dice'

    print('epoch |    lr    |      %        |  loss  |  avg   |  dloss |  dice  |  best  | time |  save |')

    if not args.no_first_val:
        val_metrics = validate(args, model, loaders['valid'])
        print('val   |          |               |        |        | {:.4f} | {:.4f} | {:.4f} |        |        |'.format(
            val_metrics['dice_loss'], val_metrics['dice'], val_metrics['dice']))

        best_metrics = val_metrics[best_key]

    if args.val:
        return

    model.train()

    #if args.lrs == 'plateau':
    #    lr_scheduler.step(best_metrics)
    #else:
    #    lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.num_epochs):
        train_loss = 0

        current_lr = get_lrs(optimizer)
        bg = time.time()
        for batch_idx, data in enumerate(loaders['train']):
            train_iter += 1
            img, targets  = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
          
            outputs = model(img)
            loss = criterion(outputs, targets)
            (loss*batch_size).backward()
            
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            print('\r {:4d} | {:.6f} | {:06d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), loaders['train'].num, 
                loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                save_model(model, model_file+'_latest')
                val_metrics = validate(args, model, loaders['valid'])
                
                _save_ckp = ''
                if val_metrics[best_key] > best_metrics:
                    best_metrics = val_metrics[best_key]
                    save_model(model, model_file)
                    _save_ckp = '*'
                print(' {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
                    val_metrics['dice_loss'], val_metrics['dice'], best_metrics,
                    (time.time() - bg) / 60, _save_ckp))

                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(best_metrics)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    #del model, optimizer, lr_scheduler

def save_model(model, model_file):
    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), model_file)
    else:
        torch.save(model.state_dict(), model_file)

def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs


def validate(args, model: nn.Module, valid_loader):
    model.eval()
    all_preds, all_targets, all_loss = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets)
            
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            all_loss.append(loss.item())
            all_preds.append(outputs.cpu())

    all_preds = torch.cat(all_preds, 0)
    all_targets = torch.cat(all_targets, 0)

    metrics = {}
    metrics['dice_loss'] = np.mean(all_loss)
    metrics['dice'] = dice(all_preds, all_targets)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--encoder_type', type=str, required=True)
    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--ckp', type=str, default=None)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--iter_val', default=400, type=int, help='start epoch')
    parser.add_argument('--num_epochs', default=60, type=int, help='epoch')
    parser.add_argument('--optim_name', default='RAdam', choices=['SGD', 'RAdam', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=3, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--ifold', default=0, type=int, help='lr scheduler patience')
    
    args = parser.parse_args()
    print(args)
    train(args)
