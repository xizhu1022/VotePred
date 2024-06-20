from collections import defaultdict as ddict

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MyData, pad_collate_mids
from utils import cal_results


class Trainer(object):
    def __init__(self, model, data: MyData, args):
        self.model = model
        self.data = data
        self.device = args.device

        self.epochs = args.epochs
        self.min_epochs = args.min_epochs
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
            graph = self.data.build_graph(cid_latest=cid + 4)
            graph = graph.to(self.device)
            # edge_index_combined = self.data.edge_index_combined.to(self.device)
            # edge_type_combined = self.data.edge_type_combined.to(self.device)
            self.data.get_dataset_mids(cidstart=cid)

            train_dataset = self.data.get_train_dataset_mids()
            val_dataset = self.data.get_val_dataset_mids()
            test_dataset = self.data.get_test_dataset_mids()

            run_results = self.run(train_dataset, val_dataset, test_dataset, graph)
            print('[Run] cid: %d, acc: %.4f, f1: %.4f, recall: %.4f, '
                  'pre: %.4f, auc: %.4f' % (cid,
                                            run_results['acc'],
                                            run_results['f1'],
                                            run_results['recall'],
                                            run_results['pre'],
                                            run_results['auc'])
                  )

            for metric, result in run_results.items():
                all_run_result[metric].append(result)

            # del graph
            # torch.cuda.empty_cache()
            # gc.collect()

        print('[Overall] acc: %.4f, f1: %.4f, recall: %.4f, pre: %.4f, auc: %.4f' % (np.mean(all_run_result['acc']),
                                                                                     np.mean(all_run_result['f1']),
                                                                                     np.mean(all_run_result['recall']),
                                                                                     np.mean(all_run_result['pre']),
                                                                                     np.mean(all_run_result['auc']))
              )

    def run(self, train_dataset, val_dataset, test_dataset, graph):
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  collate_fn=pad_collate_mids, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                collate_fn=pad_collate_mids, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                 collate_fn=pad_collate_mids, pin_memory=True)

        best_val_metric, best_val_epoch = 0., 0
        best_test_metric, best_test_epoch = 0., 0
        best_val_result, best_test_result = {}, {}

        for epoch in range(self.epochs):
            # train
            train_loss, train_results = self.train_one_epoch(train_loader, graph)
            if (epoch + 1) % 1 == 0:
                print('[Train] epoch: %d, loss: %.4f, acc: %.4f, f1: %.4f, recall: %.4f, '
                      'pre: %.4f, auc: %.4f' % (epoch,
                                                train_loss,
                                                train_results['acc'],
                                                train_results['f1'],
                                                train_results['recall'],
                                                train_results['pre'],
                                                train_results['auc'])
                      )
            # validate
            val_loss, val_results = self.evaluate(val_loader, graph)
            if (epoch + 1) % 5 == 0:
                print('[Valid] epoch: %d, loss: %.4f, acc: %.4f, f1: %.4f, recall: %.4f, '
                      'pre: %.4f, auc: %.4f' % (epoch,
                                                val_loss,
                                                val_results['acc'],
                                                val_results['f1'],
                                                val_results['recall'],
                                                val_results['pre'],
                                                val_results['auc'])
                      )

            val_metric = val_results['f1']
            if epoch > self.min_epochs:
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_val_epoch = epoch
                    best_val_result = val_results
                if epoch - best_val_epoch > self.patience:
                    print('Stop at epoch %d' % (epoch + 1))
                    break

        # test
        test_loss, test_results = self.evaluate(test_loader, graph)

        return test_results

    def train_one_epoch(self, train_loader, graph):
        full_targets, full_predictions, full_predicted_labels = [], [], []
        total_loss = 0
        total_count = 0
        self.model.train()

        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            loss, targets, predictions, predicted_labels = self.model(batch, graph)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().cpu().item() * len(batch)
            total_count += len(batch)
            full_targets = full_targets + targets.tolist()
            full_predictions = full_predictions + predictions.tolist()
            full_predicted_labels = full_predicted_labels + predicted_labels.tolist()

        train_results = cal_results(predictions=full_predictions,
                                    predicted_labels=full_predicted_labels,
                                    targets=full_targets)
        loss = total_loss / total_count

        return loss, train_results

    @torch.no_grad()
    def evaluate(self, loader, graph):
        full_targets, full_predictions, full_predicted_labels = [], [], []
        total_loss = 0
        total_count = 0

        self.model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            loss, targets, predictions, predicted_labels = self.model(batch, graph)

            total_loss += loss.detach().cpu().item() * len(batch)
            total_count += len(batch)
            full_targets = full_targets + targets.tolist()
            full_predictions = full_predictions + predictions.tolist()
            full_predicted_labels = full_predicted_labels + predicted_labels.tolist()

        eval_results = cal_results(predictions=full_predictions,
                                   predicted_labels=full_predicted_labels,
                                   targets=full_targets)
        loss = total_loss / total_count

        return loss, eval_results
