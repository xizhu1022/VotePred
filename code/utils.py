import random
from collections import defaultdict
import os
import dgl
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False


def cal_results(predictions, predicted_labels, targets):
    correct = np.sum(np.array(predicted_labels) == np.array(targets))

    acc = correct / len(targets)
    f1 = f1_score(targets, predicted_labels)
    recall = recall_score(targets, predicted_labels)
    pre = precision_score(targets, predicted_labels)
    auc = roc_auc_score(targets, predictions)

    results = dict()
    results['acc'] = acc
    results['f1'] = f1
    results['recall'] = recall
    results['pre'] = pre
    results['auc'] = auc

    return results


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


def generate_G_from_H(H):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    DV += 1e-12
    DE += 1e-12
    invDE = np.mat(np.diag(np.power(DE, -1)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    # if type  == "sym":
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    G = DV2 * H * W * invDE * HT * DV2  # sym
    # elif type == "asym":
    #     DV1 = np.mat(np.diag(np.power(DV, -1)))
    #     G = DV1 * H * W * invDE * HT  # asym
    # else:
    #     raise NotImplementedError
    return G


def matrix2dict(matrix: sp.csr_matrix):
    result = defaultdict(list)
    for i, j in zip(matrix.nonzero()[0], matrix.nonzero()[1]):
        result[i].append(j)
    return result


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)