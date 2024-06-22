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
from utils import seed_everything
from loguru import logger

from data import MyData
from model import RGCN_DualAttn_FFNN
from train import Trainer
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--data_path', type=str, default='../data_0609', help='data path')
    parser.add_argument('--model_path', type=str, default='../saves', help='model path')
    parser.add_argument('--log_path', type=str, default='../log', help='model path')
    parser.add_argument('--gpu', type=int, default=1, help='gpu_id')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')  # 1e-4
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--min_epochs', type=int, default=30, help='minimum epochs')
    parser.add_argument('--patience', type=int, default=20, help='patience')
    parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
    parser.add_argument('--num_layers', type=int, default=3, help='num_layers')
    parser.add_argument('--dropout_1', type=float, default=0.05, help='dropout')
    parser.add_argument('--dropout_2', type=float, default=0.05, help='dropout')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='negative_slope')
    parser.add_argument('--lambda_1', type=float, default=0.05, help='lambda_1')
    parser.add_argument('--lambda_2', type=float, default=0.05, help='lambda_2')
    parser.add_argument('--alpha', type=float, default=0.05, help='alpha')
    parser.add_argument('--train_batch_size', type=int, default=64, help='train_batch_size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='test_batch_size')
    parser.add_argument('--embedding_size', type=int, default=64, help='embedding_size')
    parser.add_argument('--model_name', type=str, default='RGCN_DualAttn_FFNN', help='model_name')

    args = parser.parse_args()

    # Device
    args.device = torch.device('cuda:{}'.format(args.gpu))

    seed_everything()

    # Dataset and Model
    myData = MyData(args.data_path)

    if args.model_name == 'RGCN_DualAttn_FFNN':
        model = RGCN_DualAttn_FFNN(
            dim=args.embedding_size,
            num_nodes=myData.num_nodes,
            num_relations=myData.num_rels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout_1=args.dropout_1,
            dropout_2=args.dropout_2,
            negative_slope=args.negative_slope,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            alpha=args.alpha,
            pretrained=myData.pretrained_embeddings,
            data=myData
        )

    # elif args.model_name == 'RGCN_Merge':
    #     model = RGCN_Merge(
    #         num_nodes=myData.num_nodes,
    #         input_size=args.embedding_size,
    #         hidden_size=args.embedding_size,
    #         output_size=args.embedding_size,
    #         num_relations=myData.num_rels
    #         )
    else:
        raise NotImplementedError

    # Log
    args.this_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    logfile = '{}_{}'.format(args.model_name, args.this_time)
    logger.add('{}/{}.log'.format(args.log_path, logfile), encoding='utf-8')

    model = model.to(args.device)

    # Train
    logger.info(args.__str__())
    logger.info('[File] save to {}.'.format(logfile))
    logger.info(model)
    trainer = Trainer(model=model,
                      data=myData,
                      args=args)
    trainer.multiple_runs()
