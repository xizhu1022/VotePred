import numpy as np
import scipy.sparse as sp
import torch


class EarlyStopping:
    def __init__(self, patience=3, delta=0, path='checkpoint_all.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, acc_score, model):
        score = acc_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc_score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Model saved! Validation acc: {val_loss:.4f}')


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    graph = graph.astype(np.float32)

    return graph


def adj_matrix_to_edge_index(adj_matrix):
    rows, cols = adj_matrix.nonzero()
    rows = torch.from_numpy(rows)
    cols = torch.from_numpy(cols)
    edge_index = torch.stack([rows, cols], dim=0)
    return edge_index
