from collections import defaultdict as ddict

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MyData, pad_collate
from model import RGCN_Merge


class Trainer(object):
    def __init__(self, model: RGCN_Merge, data: MyData, args):
        self.model = model
        self.data = data
        self.device = args.device

        self.epochs = args.epochs
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.model_path = args.model_path
        self.lr = args.lr
        self.patience = args.patience

        self.cidstart_list = self.data.cidstart_list
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def multiple_runs(self):
        all_run_result = ddict(list)

        for cid in self.cidstart_list:
            print('------------{}-------------'.format(cid))
            self.data.get_dataset_vids(cidstart=cid)
            train_dataset = self.data.get_train_dataset()
            val_dataset = self.data.get_val_dataset()
            test_dataset = self.data.get_test_dataset()

            run_result = self.run(train_dataset, val_dataset, test_dataset)
            print('[Run] cid: %d, acc: %.4f, f1: %.4f, recall: %.4f, pre: %.4f' % (cid,
                                                                                   run_result['acc'], run_result['f1'],
                                                                                   run_result['recall'],
                                                                                   run_result['pre'])
                  )

            for metric, result in run_result.items():
                all_run_result[metric].append(result)

        print('[Overall] acc: %.4f, f1: %.4f, recall: %.4f, pre: %.4f' % (np.mean(all_run_result['acc']),
                                                                          np.mean(all_run_result['f1']),
                                                                          np.mean(all_run_result['recall']),
                                                                          np.mean(all_run_result['pre']))
              )

    def run(self, train_dataset, val_dataset, test_dataset):
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  collate_fn=pad_collate, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                collate_fn=pad_collate, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                 collate_fn=pad_collate, pin_memory=True)

        best_val_metric, best_val_epoch = 0., 0
        best_test_metric, best_test_epoch = 0., 0
        best_val_result, best_test_result = {}, {}

        for epoch in range(self.epochs):
            # train
            train_result = self.train_one_epoch(train_loader, epoch)
            if (epoch + 1) % 1 == 0:
                print('[Train] epoch: %d, acc: %.4f, f1: %.4f, recall: %.4f, pre: %.4f' % (epoch,
                                                                                           train_result['acc'],
                                                                                           train_result['f1'],
                                                                                           train_result['recall'],
                                                                                           train_result['pre'])
                      )
            # validate
            val_result = self.evaluate(val_loader)
            if (epoch + 1) % 1 == 0:
                print('[Valid] epoch: %d, acc: %.4f, f1: %.4f, recall: %.4f, pre: %.4f' % (epoch,
                                                                                           val_result['acc'],
                                                                                           val_result['f1'],
                                                                                           val_result['recall'],
                                                                                           val_result['pre'])
                      )
            val_metric = val_result['f1']
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_val_epoch = epoch
                best_val_result = val_result
            if epoch - best_val_epoch > self.patience:
                print('Stop at epoch %d' % (epoch + 1))
                break

        # test
        test_result = self.evaluate(test_loader)
        # print('[Test] epoch: %d, acc: %.4f, f1: %.4f, recall: %.4f, pre: %.4f' % (epoch,
        #                                                                           test_result['acc'],
        #                                                                           test_result['f1'],
        #                                                                           test_result['recall'],
        #                                                                           test_result['pre'])
        #       )
        # test_metric = test_result['f1']
        # if test_metric > best_test_metric:
        #     best_test_metric = test_metric
        #     best_test_epoch = epoch
        #     best_test_result = test_result

        return test_result

    def train_one_epoch(self, train_loader, epoch):
        train_result = ddict(list)
        self.model.train()

        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            bce_loss, acc, f1, recall, pre, auc = self.model(batch)
            loss = bce_loss
            loss.backward()
            self.optimizer.step()

            loss_scalar = loss.detach()

            train_result['acc'].append(acc)
            train_result['f1'].append(f1)
            train_result['recall'].append(recall)
            train_result['pre'].append(pre)
        for k, v in train_result.items():
            train_result[k] = np.mean(v)
        return train_result

    @torch.no_grad()
    def evaluate(self, loader):
        eval_result = ddict(list)
        self.model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            bce_loss, acc, f1, recall, pre, auc = self.model(batch)
            eval_result['acc'].append(acc)
            eval_result['f1'].append(f1)
            eval_result['recall'].append(recall)
            eval_result['pre'].append(pre)
        for k, v in eval_result.items():
            eval_result[k] = np.mean(v)
        return eval_result

