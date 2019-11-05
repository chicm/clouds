import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,Dataset
from loader import CloudDataset, get_train_val_loaders, get_test_loader
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, InferCallback, CheckpointCallback
from models import create_model
from tqdm import tqdm
import cv2
from utils import post_process, sigmoid, dice, get_validation_augmentation, get_preprocessing, mask2rle
import segmentation_models_pytorch as smp
import ttach as tta
import settings

#w = np.array([1., 0.5, 0.5, 0.5, 0.5, 0.3, 1, 1, 0.5])
#w = np.array([1.]*5)
#w = np.array([1.,1.,1.,1.,1.])
#w /= w.sum()
#print(w)

def create_models(args):
    models = []
    for encoder_type, ckp in zip(args.encoder_types.split(','), args.ckps.split(',')):
        model = create_model(encoder_type, ckp=ckp, act='sigmoid')[0].cuda()
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
        models.append(model)
    return models

def predict_loader(models, loader):
    probs, masks = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            img, mask = batch[0].cuda(), batch[1]
            masks.append(mask)
            outputs = []
            for i, model in enumerate(models):
                output = model(img).cpu() #** 0.5
                outputs.append(output)
            avg_ouput = torch.stack(outputs).mean(0)
            probs.append(avg_ouput)
    probs = torch.cat(probs, 0).numpy()
    masks = torch.cat(masks, 0).numpy()
    print('probs:', probs.shape)
    print('masks:', masks.shape)
    return probs, masks

def ensemble(args):
    #class_params = {0: (0.5, 25000), 1: (0.7, 15000), 2: (0.4, 25000), 3: (0.6, 10000)}
    models = create_models(args)
    class_params = find_class_params(args, models)
    #exit(0)

    test_loader = get_test_loader(args.encoder_types.split(',')[0], args.batch_size)
    probs, _ = predict_loader(models, test_loader)

    encoded_pixels, encoded_pixels_no_minsize = [], []
    image_id = 0
    for img_out in tqdm(probs):
        #runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
        #for i, batch in enumerate(runner_out):
        for probability in img_out:
            
            #probability = probability.cpu().detach().numpy()
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(probability, class_params[image_id % 4][0], class_params[image_id % 4][1])
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)

            predict2, num_predict2 = post_process(probability, class_params[image_id % 4][0], 0)
            if num_predict2 == 0:
                encoded_pixels_no_minsize.append('')
            else:
                r2 = mask2rle(predict2)
                encoded_pixels_no_minsize.append(r2)

            image_id += 1

    sub = pd.read_csv(os.path.join(settings.DATA_DIR, 'sample_submission.csv'))

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(args.out, columns=['Image_Label', 'EncodedPixels'], index=False)

    sub['EncodedPixels'] = encoded_pixels_no_minsize
    sub.to_csv(args.out+'_no_minsize', columns=['Image_Label', 'EncodedPixels'], index=False)


def find_class_params(args, models):
    val_loader = get_train_val_loaders(args.encoder_types.split(',')[0], batch_size=args.batch_size)['valid']
    probs, masks = predict_loader(models, val_loader)
    print(probs.shape, masks.shape)

    valid_masks = []
    probabilities = np.zeros((2220, 350, 525))
    for i, (img_probs, img_masks) in enumerate(zip(probs, masks)):
        for m in img_masks:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(img_probs):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability

    print(len(valid_masks), len(probabilities), valid_masks[0].shape, probabilities[0].shape)
    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(30, 90, 5):
            t /= 100
            #for ms in [0, 100, 1200, 5000, 10000]:
            for ms in [5000, 10000, 15000, 20000, 22500, 25000]:
            
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(probability, t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])


        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        
        class_params[class_id] = (best_threshold, best_size)
    print(class_params)
    return class_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--encoder_types', type=str, required=True)
    parser.add_argument('--ckps', type=str, required=True)
    parser.add_argument('--out', type=str, default='ensemble.csv')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--ifold', default=0, type=int, help='lr scheduler patience')
    
    args = parser.parse_args()
    print(args)
    ensemble(args)
    #find_class_params(args)
    #tmp = torch.load(args.ckp)
    #print(tmp.keys())
    #print(tmp['valid_metrics'])
