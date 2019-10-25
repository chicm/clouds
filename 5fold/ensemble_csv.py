import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def ensemble(args):
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--dfs', type=str, required=True)
    parser.add_argument('--out', type=str, default='ensemble.csv')
    
    args = parser.parse_args()
    print(args)
    ensemble(args)
