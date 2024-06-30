import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch


data_path = '../data_0609'
path1 = '../tsne_data'
group2nodes = np.load(os.path.join(path1, 'group_nodes_dict_115.npy'), allow_pickle=True).item()
all_nodes = np.load(os.path.join(path1, 'm_node_list_115.npy'), allow_pickle=True).tolist()
vid_g_rate = np.load(os.path.join(path1, 'vid_group_supportrate_dict.npy'), allow_pickle=True).item()

bill2name = np.load(os.path.join(data_path, 'node_vid_115_dict.npy'), allow_pickle=True).item()
bill2results = np.load(os.path.join(data_path, 'index_results_dict.npy'), allow_pickle=True).item()
ori_node2index = np.load(os.path.join(data_path, 'node_index_dict.npy'), allow_pickle=True).item()
ori_index2node = {index: node for node, index in ori_node2index.items()}

cid = 112
path2 = '../final_saves'
weight_path = os.path.join(path2, str(cid), 'RGCN_DualAttn_FFNN_embeddings.pth')
weights = torch.load(weight_path)
name = 'final'

name2bill = {name: bill for bill, name in bill2name.items()}
bill_name = 'h463-115.2018'
bill_index = name2bill[bill_name]

all_nodes = all_nodes + [bill_index]
node2index = {node: index for index, node in enumerate(all_nodes)}

group2info = dict()
label_list = ['FL', 'CA', 'MI', 'NY', 'TX', 'IA']
# ['FL', 'CA', 'MI', 'NY', 'OH', 'TX', 'IN','AR','IA','KS','KY','MA']
group_list = []

for label in label_list:
    group_list.append([node2index[_] for _ in group2nodes[label]])
group2info['state'] = dict()
group2info['state']['label_list'] = label_list
group2info['state']['group_list'] = group_list

label_list = ['United States House Committee on Agriculture', 'United States House Committee on Rules', 'United States House Committee on House Administration']
group_list = []

for label in label_list:
    group_list.append([node2index[_] for _ in group2nodes[label]])
group2info['committee'] = dict()
group2info['committee']['label_list'] = label_list
group2info['committee']['group_list'] = group_list


label_list = ['D', 'R']
group_list = []
for label in label_list:
    group_list.append([node2index[_] for _ in group2nodes[label]])
group2info['party'] = dict()
group2info['party']['label_list'] = label_list
group2info['party']['group_list'] = group_list

label_list = ['yeas', 'nays']
group_list = []
for label in label_list:
    result = bill2results[bill_index][label]
    group_list.append([node2index[_] for _ in result])
group2info['vote'] = dict()
group2info['vote']['label_list'] = label_list
group2info['vote']['group_list'] = group_list

label_list = [bill_name]
group_list = []
for label in label_list:
    group_list.append([node2index[bill_index]])
group2info['bill'] = dict()
group2info['bill']['label_list'] = label_list
group2info['bill']['group_list'] = group_list

for group, info in group2info.items():
    label_list, group_list = info['label_list'], info['group_list']
    all_labels = []
    for i, i_group in enumerate(group_list):
        all_labels += (np.ones(len(i_group)) * i).tolist()
    group2info[group]['all_labels'] = all_labels

all_node_embeddings = weights[torch.LongTensor(all_nodes)]


def plot_2d(all_node_embeddings, group2info, name):
    tsne = TSNE(n_components=2,
                init='pca',
                perplexity=8, # 30
                early_exaggeration=5, # 12
                random_state=42)
    X_tsne = tsne.fit_transform(all_node_embeddings)

    # 归一化
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm= (X_tsne - x_min) / (x_max - x_min)
    font_size = 7
    # 绘制t-SNE可视化图
    # plt.figure(figsize=(7, 5), facecolor='white')
    plt.figure(figsize=(5, 5.5), facecolor='white')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 图中文字体设置为Times New Roman

    group2mark = {
        'state': 's',
        'party': '+',
        'vote': '.',
        'committee': '^'
    }
    group2size = {
        'state': 50,
        'party': 100,
        'committee': 50,
        'vote': 10
    }
    group2alpha = {
        'state': 0.8,
        'party': 0.8,
        'committee': 0.8,
        'vote': 0.4
    }
    group2color = {
        'state': ["#D76364", "#F1D77E", "#5F97D2","#9394E7","#B1CE46","#63E398", "#B9db57", '#00FFFF'],
        'committee':[ '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#FFFF00', '#FFA500', '#800080', '#00FF00', '#008080', '#FFC0CB', '#40E0D0', '#E6E6FA', '#800000', '#000080', '#808000', '#87CEEB', '#4B0082', '#FFD700', '#C0C0C0', '#228B22' ],#
        'party': ["#2F7FC1", "#D8383A"],
        'vote': ["#96C37D", "#FF8884"],
    }

    label_map = {
        'United States House Committee on Agriculture': 'Agriculture',
        'United States House Committee on Rules': 'Rules',
        'United States House Committee on House Administration': 'Administration'
    }

    for group, info in group2info.items():
        label_list, group_list, all_labels = info['label_list'], info['group_list'], np.array(info['all_labels'])
        for i, label in enumerate(label_list):
            i_group_list = np.array(group_list[i])
            if group == 'vote':
                x = X_norm[i_group_list, 0]
                y = X_norm[i_group_list, 1]
                plt.scatter(x, y, color=group2color[group][i], marker=group2mark[group], s=group2size[group],
                            label=label, alpha=group2alpha[group])
            elif group == 'bill':
                x = X_norm[i_group_list, 0]
                y = X_norm[i_group_list, 1]
                plt.scatter(x, y, color="#000000", marker='*', s=100, label=bill_name, alpha=1)
            elif group == 'committee':
                x = np.mean(X_norm[i_group_list, 0])
                y = np.mean(X_norm[i_group_list, 1])
                plt.scatter(x, y, color=group2color[group][i], marker=group2mark[group], s=group2size[group],
                            label=label_map[label] + ' ({:.2f} %)'.format(vid_g_rate[bill_name][label] * 100),
                            alpha=group2alpha[group])
            else:
                x = np.mean(X_norm[i_group_list, 0])
                y = np.mean(X_norm[i_group_list, 1])
                plt.scatter(x, y, color=group2color[group][i], marker=group2mark[group], s=group2size[group],
                            label=label + ' ({:.2f} %)'.format(vid_g_rate[bill_name][label] * 100),
                            alpha=group2alpha[group])

    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order
    order = [13, 1, 2, 8,9,11,0,3,6,10,12,5,4,7]

    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=font_size, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    ax = plt.gca()
    # ax.spines['right'].set_visible(False)  # 取消右边界
    # ax.spines['top'].set_visible(False)    # 取消上边界
    ax.spines['right'].set_linewidth('1.0')
    ax.spines['top'].set_linewidth('1.0')
    ax.spines['bottom'].set_linewidth('1.0')
    ax.spines['left'].set_linewidth('1.0')

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # plt.title(f'{name}_2d', fontsize=font_size + 2)

    # plt.show()  # 显示图形
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig(f"./figures/{name}_2d.pdf", format="pdf",dpi = 300, bbox_inches='tight', pad_inches=0.0)

plot_2d(all_node_embeddings, group2info, name)