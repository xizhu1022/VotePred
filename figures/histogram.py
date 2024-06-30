from matplotlib import pyplot as plt
import numpy as np
import os

from collections import defaultdict as ddict

path = "./ablation_studies"
if not os.path.exists(path):
    os.makedirs(path)

data = ddict(ddict)
ylims = ddict(ddict)

data['HR']['Tmall'] = [0.1431, 0.1427, 0.1423, 0.1458]
data['NDCG']['Tmall'] = [0.0821, 0.0809, 0.0806, 0.0841]

data['HR']['Taobao'] = [ 0.1364, 0.1350, 0.1326, 0.1597]
data['NDCG']['Taobao'] = [0.0916, 0.0905, 0.0886, 0.1029]

ylims['HR']['Tmall'] = [0.1410, 0.1470]
ylims['NDCG']['Tmall'] = [0.0790, 0.0850]

ylims['HR']['Taobao'] = [ 0.1240, 0.1650]
ylims['NDCG']['Taobao'] = [0.08, 0.1100]

for metric in ['HR', 'NDCG']:
    for dataset in ['Tmall', 'Taobao']:
        name = f"{metric}_{dataset}"
        x = ["w/ GPD", "w/ EPD", "w/ CPD", "Full"]
        y = data[metric][dataset]
        fig, ax=plt.subplots()
        ax.set_ylim(ymin=ylims[metric][dataset][0], ymax=ylims[metric][dataset][1])
        # ax.set_ylabel(f'{metric}@10', fontdict={'fontsize': 18})
        cmap = plt.colormaps["tab20c"]
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        colors = np.arange(4) * 4 + 2
        inner_colors = cmap(colors)
        ln = ax.bar(x, y, color=inner_colors)
        plt.title(f'{metric}@10 on {dataset} Dataset', fontdict={'fontsize': 16})
        plt.bar_label(ln, label_type='edge', padding=2, fontsize=16)

        plt.savefig(os.path.join(path, f'{name}.pdf'), format="pdf", bbox_inches='tight',pad_inches=0.0)
        plt.show()
