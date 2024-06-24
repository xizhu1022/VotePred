import argparse
import time

import torch
from loguru import logger

from data import MyData
from model import RGCN_DualAttn_FFNN
from train import Trainer
from utils import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)

    # paths
    parser.add_argument('--model_name', type=str, default='RGCN_DualAttn_FFNN', help='model_name')
    parser.add_argument('--data_path', type=str, default='../data_0609', help='data path')
    parser.add_argument('--model_path', type=str, default='../saves', help='model path')
    parser.add_argument('--log_path', type=str, default='../log', help='model path')

    # data
    parser.add_argument('--if_cold_start', action='store_true', help='if_cold_start')

    # model
    parser.add_argument('--embedding_size', type=int, default=64, help='embedding_size')
    parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
    parser.add_argument('--num_layers', type=int, default=3, help='num_layers')
    parser.add_argument('--dropout_1', type=float, default=0.05, help='dropout')
    parser.add_argument('--dropout_2', type=float, default=0.05, help='dropout')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='negative_slope')
    parser.add_argument('--lambda_1', type=float, default=0.05, help='lambda_1')
    parser.add_argument('--lambda_2', type=float, default=0.05, help='lambda_2')
    parser.add_argument('--alpha', type=float, default=0.05, help='alpha')
    parser.add_argument('--encoder_type', type=str, default='hgb', help='encoder type')
    parser.add_argument('--fusion_type', type=str, default='concat2_self_attn_mlp', help='fusion type')

    # train
    parser.add_argument('--gpu', type=int, default=1, help='gpu_id')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')  # 1e-4
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--min_epochs', type=int, default=30, help='minimum epochs')
    parser.add_argument('--patience', type=int, default=20, help='patience')
    parser.add_argument('--train_batch_size', type=int, default=64, help='train_batch_size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='test_batch_size')
    parser.add_argument('--if_pre_train', action='store_true', help='if_pre_train')
    parser.add_argument('--metric', type=str, default='loss', help='metric for validation')

    args = parser.parse_args()

    # Device
    args.device = torch.device('cuda:{}'.format(args.gpu))

    seed_everything()

    # Dataset and Model
    myData = MyData(data_path=args.data_path,
                    if_cold_start=args.if_cold_start)

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
            if_pre_train=args.if_pre_train,
            fusion_type=args.fusion_type,
            encoder_type=args.encoder_type,
            data=myData
        )
        model = model.to(args.device)

    else:
        raise NotImplementedError

    # Log
    args.this_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    logfile = '{}_{}'.format(args.model_name, args.this_time)
    logger.add('{}/{}.log'.format(args.log_path, logfile), encoding='utf-8')

    # Train
    logger.info(args.__str__())
    logger.info('[File] save to {}.'.format(logfile))
    logger.info(model)
    trainer = Trainer(model=model,
                      data=myData,
                      args=args)
    # trainer.multiple_runs()
    trainer.load_model(102, '../saves/2024-06-25_03_17_15/102/RGCN_DualAttn_FFNN.pth')
