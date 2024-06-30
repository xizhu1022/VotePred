from matplotlib import pyplot as plt
import numpy as np
import os

save_path = "param.lambda"

if not os.path.exists(save_path):
    os.makedirs(save_path)

Bei_HR = [0.0858, 0.0874, 0.0864, 0.0848, 0.0893, 0.0889, 0.0868, 0.0819,
          0.0811, 0.0799, 0.0790, 0.0791, 0.0783, 0.0787, 0.0785, 0.0780]
Bei_NDCG = [0.0434, 0.0439, 0.0433, 0.0429, 0.0451, 0.0449, 0.0441, 0.0416,
            0.0414, 0.0408, 0.0404, 0.0404, 0.0400, 0.0399, 0.0398, 0.0398]
Tmall_HR = [0.1341, 0.1372, 0.1443, 0.1450, 0.1458, 0.1449, 0.1442, 0.1442,
            0.1428, 0.1414, 0.1416, 0.1414, 0.1368, 0.1396, 0.1379, 0.1393]
Tmall_NDCG = [0.0775, 0.0787, 0.0835, 0.0839, 0.0841, 0.0831, 0.0826, 0.0823,
              0.0818, 0.0813, 0.0812, 0.0808, 0.0788, 0.0798, 0.0794, 0.0792]
Taobao_HR = [0.1347, 0.1494, 0.1556, 0.1594, 0.1597, 0.1595, 0.1576, 0.1576,
             0.1570, 0.1541, 0.1535, 0.1525, 0.1517, 0.1502, 0.1493, 0.1493]
Taobao_NDCG = [0.0887, 0.0982, 0.1014, 0.1030, 0.1029, 0.1024, 0.1014, 0.1007,
               0.0998, 0.0986, 0.0980, 0.0972, 0.0964, 0.0956, 0.0950, 0.0944]
x = np.arange(0,3.01,0.2)

datas = [ ("Beibei", Bei_HR, Bei_NDCG), ("Tmall", Tmall_HR, Tmall_NDCG), ("Taobao", Taobao_HR, Taobao_NDCG)]

ylims = [[[0.076, 0.090], [0.039, 0.046]], [[0.133, 0.147], [0.077, 0.085]], [[0.133, 0.162], [0.088, 0.104]]]

for index, d in enumerate(datas):
    name = d[0]
    hr = d[1]
    ndcg = d[2]
    ylim = ylims[index]
    fig, ax1 = plt.subplots()
    plt.xticks(fontsize=14)

    ax1.tick_params(labelsize=14)

    plt.grid(axis="both")
    ax1.set_ylim(ymin=ylim[0][0], ymax=ylim[0][1])
    ln1 = ax1.plot(x, hr, alpha = 0.4, linewidth=2, color="blue", marker="o", markersize=10)
    ax1.set_ylabel("HR@10", fontdict={'fontsize': 16})
    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=14)

    ax2.set_ylabel("NDCG@10", fontdict={'fontsize': 16})
    ax2.set_ylim(ymin=ylim[1][0], ymax=ylim[1][1])
    ln2 = ax2.plot(x, ndcg, alpha = 0.4, linewidth=2, color="red", markersize=10, marker="D")
    plt.legend(ln1 + ln2, ["HR@10", "NDCG@10"], fontsize=16)
    plt.title(f'HR@10 and NDCG@10 on {name} Dataset', fontdict={'fontsize': 16})
    # plt.show()
    plt.savefig(os.path.join(save_path, f"{name}.pdf"), format="pdf", bbox_inches='tight',pad_inches=0.0)
    plt.show()
