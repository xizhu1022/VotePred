
import matplotlib.pyplot as plt
import numpy as np
import os
# import seaborn as sns

xdim = ["0.3", "0.4", "0.5", "0.6","0.7"]
ydim = ["0.1", "0.2", "0.3","0.4", "0.5"]


f1_data = np.array([[0.6596, 0.6598, 0.6805, 0.6628, 0.673],
[0.6634,0.6798,0.6821,0.6499,0.6837],
[0.6505,0.6747,0.6898,0.6781,0.676],
[0.6417,0.6802,0.669,0.6842,0.6746],
 [0.6537,0.6864,0.6739,0.6741,0.6727]])

acc_data = np.array([[0.5934, 0.5755, 0.5978, 0.593, 0.5906],
[0.6018, 0.6172, 0.6006, 0.5851, 0.5945],
[0.5739, 0.6015, 0.6251, 0.6147, 0.6081],
[0.5682, 0.6204, 0.6077, 0.6113, 0.6192],
[0.5908, 0.6247, 0.606, 0.6138, 0.597]])

data = acc_data
plt.figure(figsize=(8, 4))
plt.xticks(np.arange(len(ydim)), labels=ydim)
           # rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(xdim)), labels=xdim)
plt.title("Accuracy")

for i in range(len(xdim)):
    for j in range(len(ydim)):
        if (i==0 and j==1) or (i==3 and j==0) or (i==2 and j==0):
            text = plt.text(j, i, data[i, j], ha="center", va="center", color="navy")
        else:
            text = plt.text(j, i, data[i, j], ha="center", va="center", color="w")

fontdict={ 'variant': 'normal', 'weight': 'light'}
plt.xlabel(r'$\lambda_1$', fontdict=fontdict)
plt.ylabel(r'$\lambda_2$', fontdict=fontdict)

path = './'
name = 'f1_heatmap'
plt.savefig(os.path.join(path, f'{name}.pdf'), format="pdf", bbox_inches='tight', pad_inches=0.0)

plt.imshow(data,
           cmap='Blues',
           aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.show()
