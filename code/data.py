import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from utils import adj_matrix_to_edge_index, matrix2dict


class MyDataset(Dataset):
    def __init__(self, vids, node_index_dict, index_results_dict, num_node):
        self.node_index_dict = node_index_dict
        self.index_results_dict = index_results_dict
        self.vids = vids
        self.num_node = num_node
        # window = cid_window_dict[cidstart]
        # self.all_vids = []
        # for cid in window[:4]:
        #     self.all_vids.append(cid_vid_dict[cid])

    def __getitem__(self, index):
        vid = self.vids[index]
        vid_index = self.node_index_dict[vid] # unsolved problem
        results = self.index_results_dict[vid_index]  # 投票结果
        candidate_pos = results["yeas"]
        candidate_neg = results['nays'] + results['absents']  # 从弃权和反对票一起负采样

        if len(candidate_pos) == 0:
            pos_index = self.num_node
        else:
            pos_index = random.sample(candidate_pos, 1)[0]
        if len(candidate_neg) == 0:
            neg_index = self.num_node +1
        else:
            neg_index = random.sample(candidate_neg, 1)[0]
        # print(vid, len(results["yeas"]), len(results["nays"]), len(results['absents']))

        return (vid_index, pos_index, neg_index)

    def __len__(self):
        return len(self.vids)

    # for cid in window[:4]:
    #     vids = self.cid_vids_dict[cid]  # 对应佳悦的 all_result_id
    #     hmids = self.cid_hmids_dict[cid]  # 对应佳悦的 candidata,佳悦好像把全量议员(1600多人)都设置成了负样本候选？ 不知道为何这么设置，我只选取了本届国会的众议员400人
    #     # hmids_index = [self.node_index_dict[node] for node in hmids]  # 所有负样本候选议员的index
    #     for vid in vids:
    #         vid_index = self.node_index_dict[vid]
    #         results = self.index_results_dict[vid_index]  # 投票结果
    #         candidate_pos = results["yeas"]
    #         candidate_neg = results['nays'] + results['absents']  # 从弃权和反对票一起负采样
    #         for pos_index in candidate_pos:
    #             neg_index = random.choice(candidate_neg)
    #             sample.append((vid_index, pos_index, neg_index))


class MyData(object):
    def __init__(self, load_path):
        self.load_path = load_path

        self.load_data()
        self.build_graph()

    def load_data(self):
        # 加载实体   法案立法者与国会届次相关, 主题委员会党派与国会届次无关
        # 国会届次 102-116
        self.cid_list = np.load(os.path.join(self.load_path, 'cid_list.npy'), allow_pickle=True).tolist()
        # 时间窗口起始点 102-112
        self.cidstart_list = np.load(os.path.join(self.load_path, 'cidstart_list.npy'), allow_pickle=True).tolist()
        # 国会届次-时间窗口
        self.cid_window_dict = np.load(os.path.join(self.load_path, 'cidstart_window_dict.npy'),
                                       allow_pickle=True).item()
        # 国会届次-法案(众)
        self.cid_vids_dict = np.load(os.path.join(self.load_path, 'cid_vids_dict.npy'), allow_pickle=True).item()
        # 国会届次-立法者(参+众)
        self.cid_mids_dict = np.load(os.path.join(self.load_path, 'cid_mids_dict.npy'), allow_pickle=True).item()
        # 国会届次-立法者(众)
        self.cid_hmids_dict = np.load(os.path.join(self.load_path, 'cid_hmids_dict.npy'), allow_pickle=True).item()
        # 国会届次-立法者(参)
        self.cid_smids_dict = np.load(os.path.join(self.load_path, 'cid_smids_dict.npy'), allow_pickle=True).item()

        # 所有实体列表 11639 = 8165(法案)+1674(立法者)+1753(主题)+44(委员会)+3(党派)
        self.node_list = np.load(os.path.join(self.load_path, 'node_list.npy'), allow_pickle=True).tolist()
        # 所有实体index
        self.node_index_dict = np.load(os.path.join(self.load_path, 'node_index_dict.npy'), allow_pickle=True).item()
        # 所有法案列表 8165
        self.vid_list = np.load(os.path.join(self.load_path, 'vid_list.npy'), allow_pickle=True).tolist()
        # 所有立法者列表 1674
        self.mid_list = np.load(os.path.join(self.load_path, 'mid_list.npy'), allow_pickle=True).tolist()
        # 所有主题列表 1753
        self.subject_list = np.load(os.path.join(self.load_path, 'subject_list.npy'), allow_pickle=True).tolist()
        # 所有委员会列表 44
        self.committee_list = np.load(os.path.join(self.load_path, 'committee_list.npy'), allow_pickle=True).tolist()
        # 所有党派列表 3 R D I
        self.party_list = np.load(os.path.join(self.load_path, 'party_list.npy'), allow_pickle=True).tolist()

        # 加载边关系领接矩阵
        self.cosponsor_network_sparse = sp.load_npz(os.path.join(self.load_path, 'cosponsor_network_sparse.npz'))
        self.subject_network_sparse = sp.load_npz(os.path.join(self.load_path, 'subject_network_sparse.npz'))
        self.committee_network_sparse = sp.load_npz(os.path.join(self.load_path, 'committee_network_sparse.npz'))
        self.twitter_network_sparse = sp.load_npz(os.path.join(self.load_path, 'twitter_network_sparse.npz'))
        self.party_network_sparse = sp.load_npz(os.path.join(self.load_path, 'party_network_sparse.npz'))

        self.bill2cosponsers = matrix2dict(self.cosponsor_network_sparse)
        self.bill2subjects = matrix2dict(self.subject_network_sparse)

        # self.cosponsor_network = np.asmatrix(cosponsor_network_sparse.toarray())  # 11639, 11639
        # self.subject_network = np.asmatrix(subject_network_sparse.toarray())
        # self.committee_network = np.asmatrix(committee_network_sparse.toarray())
        # self.twitter_network = np.asmatrix(twitter_network_sparse.toarray())
        # self.party_network = np.asmatrix(party_network_sparse.toarray())

        # 加载label相关
        self.vid_results_dict = np.load(os.path.join(self.load_path, 'vid_results_dict.npy'),
                                        allow_pickle=True).item()  # 法案-投票记录
        self.index_results_dict = np.load(os.path.join(self.load_path, 'index_results_dict.npy'),
                                          allow_pickle=True).item()  # 法案-投票记录(实体id替换为 node_list的index)


    def build_graph(self):
        edge_index0 = adj_matrix_to_edge_index(self.cosponsor_network)
        edge_index1 = adj_matrix_to_edge_index(self.subject_network)
        edge_index2 = adj_matrix_to_edge_index(self.committee_network)
        edge_index3 = adj_matrix_to_edge_index(self.twitter_network)
        edge_index4 = adj_matrix_to_edge_index(self.party_network)

        edge_type0 = torch.zeros(edge_index0.size(1), dtype=torch.long)
        edge_type1 = torch.ones(edge_index1.size(1), dtype=torch.long)
        edge_type2 = 2 * torch.ones(edge_index2.size(1), dtype=torch.long)
        edge_type3 = 3 * torch.ones(edge_index3.size(1), dtype=torch.long)
        edge_type4 = 4 * torch.ones(edge_index4.size(1), dtype=torch.long)

        self.edge_index_combined = torch.cat([edge_index0, edge_index1, edge_index2, edge_index3, edge_index4], dim=1)
        self.edge_type_combined = torch.cat([edge_type0, edge_type1, edge_type2, edge_type3, edge_type4])

    def get_dataset_vids(self, cidstart):
        # train data
        window = self.cid_window_dict[cidstart]
        self.train_vids = []
        for cid in window[:4]:
            vids = self.cid_vids_dict[cid]
            self.train_vids += vids

        # val / test data
        cid = cidstart + 4
        vids = self.cid_vids_dict[cid]
        random.shuffle(vids)
        val_size = len(vids) // 2
        self.val_vids, self.test_vids = vids[:val_size], vids[val_size:]

    def get_train_dataset(self):
        return MyDataset(vids=self.train_vids,
                         node_index_dict=self.node_index_dict,
                         index_results_dict=self.index_results_dict,
                         num_node = len(self.node_list))

    def get_val_dataset(self):
        return MyDataset(vids=self.val_vids,
                         node_index_dict=self.node_index_dict,
                         index_results_dict=self.index_results_dict,
                         num_node = len(self.node_list))

    def get_test_dataset(self):
        return MyDataset(vids=self.test_vids,
                         node_index_dict=self.node_index_dict,
                         index_results_dict=self.index_results_dict,
                         num_node = len(self.node_list))

    # def get_dataloader(self):
    #     # 计算数据集的长度和对应的切分点
    #     val_ratio = 0.5
    #     valtest_size = len(self.dataset_test)
    #     val_size = int(val_ratio * valtest_size)
    #
    #     # 创建随机采样器
    #     indices = list(range(valtest_size))
    #     val_indices = indices[:val_size]
    #     test_indices = indices[val_size:]
    #
    #     val_sampler = SubsetRandomSampler(val_indices)
    #     test_sampler = SubsetRandomSampler(test_indices)
    #
    #     # 创建DataLoader对象
    #     train_loader = DataLoader(self.dataset_train, batch_size=4096, num_workers=10, shuffle=True)
    #     val_loader = DataLoader(self.dataset_test, batch_size=4096, sampler=val_sampler, num_workers=10)
    #     test_loader = DataLoader(self.dataset_test, batch_size=4096, sampler=test_sampler, num_workers=10)
    #
    #     return train_loader, val_loader, test_loader
