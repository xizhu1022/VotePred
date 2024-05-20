import os
import random
import argparse
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from data import MyData
from model import rgcn_bert
from train import Trainer
from time import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--data_path', type=str, default='../data_0510', help='data path')
    parser.add_argument('--model_path', type=str, default='../saves', help='model path')
    parser.add_argument('--gpu', type=int, default=1, help='gpu_id')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--train_batch_size', type=int, default=128, help='train_batch_size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='test_batch_size')

    args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu))

    # Dataset and Model
    start = time()
    myData = MyData(args.data_path)
    model = rgcn_bert(num_users=len(myData.node_list) + 2,
                      embedding_size=64,
                      hidden_size=128,
                      output_size=64
                      )
    model = model.to(args.device)

    # Train
    trainer = Trainer(model=model,
                      data=myData,
                      args=args
                      )
    trainer.multiple_runs()
    print('[Time] total: %4f' % (time()-start))



