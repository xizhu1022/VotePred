import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from collections import defaultdict

from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False


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


def cal_loss(pos_pre, neg_pre):
    # pred: [bs, 1+neg_num]
    targets = torch.cat([torch.ones_like(pos_pre), torch.zeros_like(pos_pre)], dim=0)
    predictions = torch.cat([pos_pre, neg_pre], dim=0)
    # predictions[predictions == torch.] = 0.
    bce_loss = F.binary_cross_entropy_with_logits(predictions, targets)
    predicted_labels = torch.round(torch.sigmoid(predictions))

    correct = (predicted_labels == targets).sum().item()
    accuracy = correct / targets.size(0)

    try:
        f1 = f1_score(targets.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        recall = recall_score(targets.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        precision = precision_score(targets.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        auc = roc_auc_score(targets.detach().cpu().numpy(), torch.sigmoid(predictions).detach().cpu().numpy())
    except:
        print(predicted_labels.detach().cpu().numpy())
        print(predictions.detach().cpu().numpy())


    return bce_loss, accuracy, f1, recall, precision, auc


def matrix2dict(matrix: sp.csr_matrix):
    result = defaultdict(list)
    for i, j in zip(matrix.nonzero()[0], matrix.nonzero()[1]):
        result[i].append(j)
    return result