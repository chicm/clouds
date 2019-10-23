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

import settings

'''
def create_runner(args):
    model = create_model(args.encoder_type, ckp=args.ckp).cuda()
    model = nn.DataParallel(model)
    runner = SupervisedRunner(model=model)

    return runner
'''
def find_class_params(args):
    runner = SupervisedRunner()
    model = create_model(args.encoder_type)
    valid_loader = get_train_val_loaders(args.encoder_type, batch_size=args.batch_size)['valid']

    encoded_pixels = []
    loaders = {"infer": valid_loader}
    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(resume=args.ckp),
            InferCallback()
        ],
    )
    print(runner.callbacks)
    valid_masks = []
    probabilities = np.zeros((2220, 350, 525))
    for i, (batch, output) in enumerate(tqdm(zip(
            valid_loader.dataset, runner.callbacks[0].predictions["logits"]))):
        image, mask = batch
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability


    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            #for ms in [0, 100, 1200, 5000, 10000]:
            for ms in [5000, 10000, 15000, 20000, 22500, 25000, 30000]:
            
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
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
    return class_params, runner

def predict(args):
    #model = create_model(args.encoder_type, ckp=args.ckp).cuda()
    #model = nn.DataParallel(model)
    #runner = SupervisedRunner(model=model)
    class_params, runner = find_class_params(args)
    #runner = create_runner(args)

    test_loader = get_test_loader(args.encoder_type, args.batch_size)

    loaders = {"test": test_loader}

    encoded_pixels = []
    image_id = 0
    for i, test_batch in enumerate(tqdm(loaders['test'])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
        for i, batch in enumerate(runner_out):
            for probability in batch:
                
                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                image_id += 1
    
    sub = pd.read_csv(os.path.join(settings.DATA_DIR, 'sample_submission.csv'))
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(args.out, columns=['Image_Label', 'EncodedPixels'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--encoder_type', type=str, required=True)
    parser.add_argument('--ckp', type=str, required=True)
    parser.add_argument('--out', type=str)
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--ifold', default=0, type=int, help='lr scheduler patience')
    
    args = parser.parse_args()
    print(args)
    predict(args)
    #find_class_params(args)
    #tmp = torch.load(args.ckp)
    #print(tmp.keys())
    #print(tmp['valid_metrics'])
