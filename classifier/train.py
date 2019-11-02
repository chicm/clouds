import os
import argparse
import numpy as np
import pandas as pd
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
from apex import amp
import ttach as tta
from models import create_model
from loader import get_train_val_loaders, get_test_loader
from radam import RAdam
import time
from tqdm import tqdm
import settings

def _reduce_loss(loss):
    #print('loss shape:', loss.shape)
    return loss.sum() / loss.shape[0]

#c = MixedLoss()
c = nn.BCEWithLogitsLoss(reduction='none')

def criterion(y_pred, y_true):
    return c(y_pred, y_true)


def train(args):
    model, model_file = create_model(args.encoder_type, work_dir=args.work_dir, ckp=args.ckp)
    model = model.cuda()

    loaders = get_train_val_loaders(batch_size=args.batch_size)

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

    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)



    best_metrics = 0.
    best_key = 'dice'

    print('epoch |    lr    |      %        |  loss  |  avg   |   loss |  dice  |  best  | time |  save |')

    if not args.no_first_val:
        val_metrics = validate(args, model, loaders['valid'])
        print('val   |          |               |        |        | {:.4f} | {:.4f} | {:.4f} |        |        |'.format(
            val_metrics['loss'], val_metrics['dice'], val_metrics['dice']))

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
            loss = _reduce_loss(criterion(outputs, targets))
            (loss).backward()

            #with amp.scale_loss(loss*batch_size, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            
            if batch_idx % 4 == 0:
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
                    val_metrics['loss'], val_metrics['dice'], best_metrics,
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

from sklearn.metrics import precision_recall_curve, auc
class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']

def validate(args, model: nn.Module, valid_loader):
    model.eval()
    if args.tta:
        #model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    all_preds, all_targets, all_loss = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets)
            
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, nn.Sigmoid())
            loss = criterion(outputs, targets).sum()
            
            all_loss.append(loss.item())
            all_preds.append(outputs.cpu())

    all_preds = torch.cat(all_preds, 0).numpy()
    all_targets = torch.cat(all_targets, 0).numpy()

    metrics = {}
    metrics['loss'] = sum(all_loss) / valid_loader.num
    #metrics['dice'] = 0. #dice(all_preds, all_targets)

    pr_auc_mean = 0.
    for class_i in range(4):
        precision, recall, _ = precision_recall_curve(all_targets[:, class_i], all_preds[:, class_i])
        pr_auc = auc(recall, precision)
        pr_auc_mean += pr_auc/4
        #print(f"PR AUC {self.class_names[class_i]}, {self.stage}: {pr_auc:.3f}\n")
    metrics['dice'] = pr_auc_mean

    if args.val:
        print_metrics(all_preds, all_targets)

    return metrics


def predict_empty_masks(args, recall_thresholds=[0.2502807, 0.30874616, 0.47154653, 0.25778872]):
    #[0.30248615, 0.4076966, 0.55904335, 0.29780537] 0875):
    #[0.32873523, 0.44805834, 0.6001048, 0.3136805] 085):
    #[0.21345863, 0.1824504, 0.41996846, 0.20917079]095): 
    # #[0.28479144, 0.35337192, 0.5124028, 0.27734384]090):
    model, model_file = create_model(args.encoder_type, work_dir=args.work_dir, ckp=args.ckp)
    model = model.cuda()
    model = DataParallel(model)
    model.eval()
    if args.tta:
        model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    test_loader = get_test_loader(batch_size=args.val_batch_size)
    all_preds = []
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = inputs.cuda()
            outputs = model(inputs, nn.Sigmoid())
            all_preds.append(outputs.cpu())
    all_preds = torch.cat(all_preds, 0).numpy()
    img_ids = test_loader.img_ids
    print(all_preds.shape)
    image_labels_empty = []
    for i, (img, predictions) in enumerate(zip(img_ids, all_preds)):
        for class_i, class_name in enumerate(class_names):
            if predictions[class_i] < recall_thresholds[class_i]:
                image_labels_empty.append(f'{img}_{class_name}')

    pd.DataFrame({'empty_ids': image_labels_empty}).to_csv('empty_ids.csv', index=False)

def print_metrics(y_pred, y_true):
    pr_auc_mean = 0.
    for class_i in range(4):
        print('class_i:', class_i)
        precision, recall, thresholds = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
        #print('precisin:', precision)
        #print('recall:', recall)
        #print('thresholds:', thresholds)
        pr_auc = auc(recall, precision)
        print('auc:', pr_auc)
        pr_auc_mean += pr_auc/4
    
    print('auc mean:', pr_auc_mean)

    best_recall_thresholds, _ = get_best_thresholds(y_pred, y_true)
    print('best recall thresholds:', best_recall_thresholds)
    print_empty_percent(best_recall_thresholds, y_pred, y_true)

def print_empty_percent(ths, y_pred, y_true):
    for i, th_i in enumerate(ths):
        percent_empty = (y_pred[:, i] < th_i).sum() / len(y_pred)
        print('empty percent:', class_names[i], percent_empty)

def get_best_thresholds(y_pred, y_true):
    recall_thresholds = []
    precision_thresholds = []
    for i in range(4):
         th = get_threshold_for_recall(y_true, y_pred, i, plot=True)
         recall_thresholds.append(th[0])
         precision_thresholds.append(th[1])

    return recall_thresholds, precision_thresholds

def get_threshold_for_recall(y_true, y_pred, class_i, recall_threshold=0.92, precision_threshold=0.90, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
    i = len(thresholds) - 1
    best_recall_threshold = None
    while best_recall_threshold is None:
        next_threshold = thresholds[i]
        next_recall = recall[i]
        if next_recall >= recall_threshold:
            best_recall_threshold = next_threshold
        i -= 1
        
    # consice, even though unnecessary passing through all the values
    best_precision_threshold = [thres for prec, thres in zip(precision, thresholds) if prec >= precision_threshold][0]

    return best_recall_threshold, best_precision_threshold


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
    parser.add_argument('--num_epochs', default=40, type=int, help='epoch')
    parser.add_argument('--optim_name', default='RAdam', choices=['SGD', 'RAdam', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=3, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--ifold', default=0, type=int, help='lr scheduler patience')
    
    args = parser.parse_args()
    print(args)
    if args.predict:
        predict_empty_masks(args)
    else:
        train(args)
