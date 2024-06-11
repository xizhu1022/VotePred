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

from data import MyData
from model import RGCN_Merge, RGCN_DualAttn_FFNN
from train import Trainer
from time import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--data_path', type=str, default='../data_0609', help='data path')
    parser.add_argument('--model_path', type=str, default='../saves', help='model path')
    parser.add_argument('--gpu', type=int, default=1, help='gpu_id')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--train_batch_size', type=int, default=16, help='train_batch_size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='test_batch_size')
    parser.add_argument('--embedding_size', type=int, default=64, help='embedding_size')
    parser.add_argument('--model_name', type=str, default='RGCN_DualAttn_FFNN', help='model_name')

    args = parser.parse_args()

    # Device
    args.device = torch.device('cuda:{}'.format(args.gpu))

    seed_everything()

    # Dataset and Model
    start = time()
    myData = MyData(args.data_path)

    if args.model_name == 'RGCN_DualAttn_FFNN':
        model = RGCN_DualAttn_FFNN(
            dim=args.embedding_size,
            graph=myData.graph.to(args.device),
            num_nodes=myData.num_nodes,
            num_relations=myData.num_rels,
            num_layers=args.num_layers,
            edge_index=myData.edge_index_combined.to(args.device),
            edge_type=myData.edge_type_combined.to(args.device),
            num_heads=args.num_heads,
            dropout=args.dropout,
            pretrained = myData.pretrained_embeddings,
            data=myData
        )

    elif args.model_name == 'RGCN_Merge':
        model = RGCN_Merge(
            num_nodes=myData.num_nodes,
            input_size=args.embedding_size,
            hidden_size=args.embedding_size,
            output_size=args.embedding_size,
            num_relations=myData.num_rels
            )
    else:
        raise NotImplementedError

    model = model.to(args.device)

    # Train
    trainer = Trainer(model=model,
                      data=myData,
                      args=args
                      )
    trainer.multiple_runs()
    print('[Time] total: %4f' % (time()-start))



