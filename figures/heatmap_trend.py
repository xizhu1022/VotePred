
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from collections import defaultdict as ddict


def find_adj(source, target, edge_weights):
    neigh_dict = ddict(int)
    num_edges = len(source)
    adjw_list = []
    for i in range(num_edges):
        s, t = source[i], target[i]
        neigh_dict[s] = neigh_dict[s] + 1
        neigh_dict[t] = neigh_dict[t] + 1
    for i in range(0, num_edges):
        s, t = source[i], target[i]
        w = edge_weights[i]
        adjw = 1 / (np.sqrt(neigh_dict[s]) * np.sqrt(neigh_dict[t])) * w
        adjw_list.append(adjw)
    return np.array(adjw_list)


def find(adjw_list, source, target, edge_types):
    node_rel_weights = ddict()
    nodes = list(set(source) | set(target))
    for nid in nodes:
        rel_dict = ddict(float)
        node_rel_weights[nid] = rel_dict
    for i in range(len(source)):
        s, t, r, adjw = source[i], target[i], edge_types[i], adjw_list[i]
        node_rel_weights[s][r] += adjw
        node_rel_weights[t][r] += adjw
    for node, node_dict in node_rel_weights.items():
        weight_sum = 0.
        for rel, weight in node_dict.items():
            # if rel in [2,3,4,5,]
            weight_sum += weight
        for r in node_rel_weights[node].keys():
            node_rel_weights[node][r] = node_rel_weights[node][r] / weight_sum
    rel_dict = ddict(float)
    for node, node_dict in node_rel_weights.items():
        for rel, weight in node_dict.items():
            rel_dict[rel] += weight
    return [rel_dict[2], rel_dict[4], rel_dict[5], rel_dict[3]]



path = '../final_saves'
results = []
# c t p s
for cid in range(102, 113):
    weight_path = os.path.join(path, str(cid), 'RGCN_DualAttn_FFNN_encoder_weights.pth')
    weights = torch.load(weight_path)

    source = np.array(weights['edge_source_nodes'])
    target = np.array(weights['edge_target_nodes'])
    edge_types = np.array(weights['edge_types'])
    edge_weights = np.array(weights['weights'])

    adjw_list = find_adj(source, target, edge_weights)

    cid_results = find(adjw_list, source, target, edge_types)
    cid_sum = np.sum(cid_results)
    cid_results = cid_results / cid_sum
    cid_results = list(cid_results)
    results.append(cid_results)
    print(cid_results)

data = np.transpose(np.array(results))
xdim = ["committee", "party","state", "following"]
ydim = np.arange(106,117)

# f1_data = np.random.randn(4, 11)


plt.figure(figsize=(13, 4))
plt.xticks(np.arange(len(ydim)), labels=ydim)
           # rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(xdim)), labels=xdim)
# plt.title('Congress Trend Analysis')

for i in range(len(xdim)):
    for j in range(len(ydim)):
        if i==1:
            text = plt.text(j, i, '%.4f'%data[i, j], ha="center", va="center", color="white")
        else:
            text = plt.text(j, i, '%.4f'%data[i, j], ha="center", va="center", color="green")

fontdict={ 'variant': 'normal', 'weight': 'light', 'size': 12}
plt.xlabel('congress', fontdict=fontdict)
# plt.ylabel('attention weight', fontdict=fontdict)

path = './'
name = 'Congress Trend Analysis'
plt.savefig(os.path.join(path, f'{name}.pdf'), format="pdf", bbox_inches='tight', pad_inches=0.0)

plt.imshow(data,
           cmap='Greens',
           aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.show()
