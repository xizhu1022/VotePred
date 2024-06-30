import os
from collections import defaultdict as ddict
from time import time

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from data import MyData, pad_collate_mids
from utils import cal_results, create_directory_if_not_exists


class Trainer(object):
    def __init__(self, model, data: MyData, args):
        self.model = model
        self.data = data
        self.device = args.device
        self.this_time = args.this_time
        self.model_name = args.model_name

        self.epochs = args.epochs
        self.min_epochs = args.min_epochs
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.model_path = args.model_path
        self.lr = args.lr
        self.patience = args.patience

        self.cidstart_list = self.data.cidstart_list
        self.metric = args.metric
        self.ascend = False
        self.get_metric_ascend()
        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def get_metric_ascend(self):
        if self.metric in ['f1', 'auc', 'pre', 'recall', 'auc']:
            self.ascend = True

    def load_model(self, cid, load_path):
        graph = self.data.build_graph(cid_latest=cid + 4)
        graph = graph.to(self.device)

        loaded_model = torch.load(load_path)
        self.model.load_state_dict(loaded_model)

    def multiple_runs(self):
        overall_start = time()
        all_run_results = ddict(list)

        for cid in self.cidstart_list:
            logger.info('------------{}-------------'.format(cid))
            run_start = time()
            graph = self.data.build_graph(cid_latest=cid + 4)
            graph = graph.to(self.device)
            self.data.get_dataset_mids(cidstart=cid)

            train_dataset = self.data.get_train_dataset_mids()
            val_dataset = self.data.get_val_dataset_mids()
            test_dataset = self.data.get_test_dataset_mids()

            run_results = self.run(cid, train_dataset, val_dataset, test_dataset, graph)
            logger.info('[Run] cid: %d, acc: %.4f, f1: %.4f, recall: %.4f, '
                        'pre: %.4f, auc: %.4f, time: %.4f' % (cid,
                                                              run_results['acc'],
                                                              run_results['f1'],
                                                              run_results['recall'],
                                                              run_results['pre'],
                                                              run_results['auc'],
                                                              time() - run_start)
                        )

            for metric, result in run_results.items():
                all_run_results[metric].append(result)

        logger.info('[Overall] acc: %.4f, f1: %.4f, recall: %.4f, pre: %.4f, '
                    'auc: %.4f, time: %.4f' % (np.mean(all_run_results['acc']),
                                               np.mean(all_run_results['f1']),
                                               np.mean(all_run_results['recall']),
                                               np.mean(all_run_results['pre']),
                                               np.mean(all_run_results['auc']),
                                               time() - overall_start)
                    )

    def run(self, cid, train_dataset, val_dataset, test_dataset, graph):
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                  collate_fn=pad_collate_mids, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.test_batch_size, shuffle=False,
                                collate_fn=pad_collate_mids, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                 collate_fn=pad_collate_mids, pin_memory=True)

        best_val_metric = 0 if self.ascend else np.inf
        best_val_epoch = 0
        test_results = None
        best_model = None

        encoder_attn_flag = True  # todo: remove after exps
        encoder_embeddings_flag = True

        for epoch in range(self.epochs):
            # train
            epoch_start = time()
            train_loss, train_results = self.train_one_epoch(train_loader, graph)
            if (epoch + 1) % 1 == 0:
                logger.info('[Train] epoch: %d, loss: %.4f, acc: %.4f, f1: %.4f, recall: %.4f, '
                            'pre: %.4f, auc: %.4f, time: %.4f' % (epoch,
                                                                  train_loss,
                                                                  train_results['acc'],
                                                                  train_results['f1'],
                                                                  train_results['recall'],
                                                                  train_results['pre'],
                                                                  train_results['auc'],
                                                                  time() - epoch_start)
                            )
            # validate
            val_loss, val_results = self.evaluate(val_loader, graph)
            val_results['loss'] = val_loss
            if (epoch + 1) % 5 == 0:
                logger.info('[Valid] epoch: %d, loss: %.4f, acc: %.4f, f1: %.4f, recall: %.4f, '
                            'pre: %.4f, auc: %.4f' % (epoch,
                                                      val_loss,
                                                      val_results['acc'],
                                                      val_results['f1'],
                                                      val_results['recall'],
                                                      val_results['pre'],
                                                      val_results['auc'])
                            )

            val_metric = val_results[self.metric]  # val_loss

            if epoch > self.min_epochs:
                if (val_metric > best_val_metric and self.ascend) or (val_metric < best_val_metric and not self.ascend):
                    # update
                    logger.info('[Valid] epoch: %d, %s: %.4f -> %.4f' % (
                        epoch,
                        self.metric,
                        best_val_metric,
                        val_metric
                    ))
                    best_val_metric = val_metric
                    best_val_epoch = epoch
                    best_val_result = val_results

                    # test
                    self.save_model(cid, self.model)
                    test_loss, test_results = self.evaluate(test_loader, graph)

                    if encoder_attn_flag and self.model.encoder_type == 'hgb':
                        weights = self.model.encoder_weights
                        self.save_weights(cid, weights)

                    if encoder_embeddings_flag:
                        embeddings = self.model.encoder_embeddings.detach().cpu().numpy()
                        self.save_embeddings(cid, embeddings)

                    logger.info('[Test] epoch: %d, loss: %.4f, acc: %.4f, f1: %.4f, recall: %.4f, '
                                'pre: %.4f, auc: %.4f' % (epoch,
                                                          test_loss,
                                                          test_results['acc'],
                                                          test_results['f1'],
                                                          test_results['recall'],
                                                          test_results['pre'],
                                                          test_results['auc'])
                                )

                if epoch - best_val_epoch > self.patience:
                    logger.info('Stop at epoch %d' % (epoch + 1))
                    break

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

    def save_model(self, cid, model):
        path = os.path.join(self.model_path, self.this_time, str(cid))
        create_directory_if_not_exists(path)
        torch.save(model.state_dict(), os.path.join(path, '{}_model.pth'.format(self.model_name)))

    def save_embeddings(self, cid, embeddings):
        path = os.path.join(self.model_path, self.this_time, str(cid))
        create_directory_if_not_exists(path)
        torch.save(embeddings, os.path.join(path, '{}_embeddings.pth'.format(self.model_name)))

    def save_weights(self, cid, weights):
        path = os.path.join(self.model_path, self.this_time, str(cid))
        saves = {
            'edge_source_nodes': self.data.edge_indexes[0].cpu().tolist(),
            'edge_target_nodes': self.data.edge_indexes[1].cpu().tolist(),
            'edge_types': self.data.edge_types.cpu().tolist(),
            'weights': weights.detach().cpu().tolist()
        }

        create_directory_if_not_exists(path)
        torch.save(saves, os.path.join(path, '{}_encoder_weights.pth'.format(self.model_name)))
