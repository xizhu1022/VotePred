import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

import dgl

from utils import adj_matrix_to_edge_index, matrix2dict


class MyDataset(Dataset):
    def __init__(self,
                 bills,
                 node2index,
                 bill2results,
                 bill2cosponsers,
                 bill2subjects_tfidf,
                 num_nodes):
        self.bills = bills  # candidate keys
        self.node2index = node2index  # node -> index
        self.bill2results = bill2results  # bill index -> voting results
        self.bill2cosponsers = bill2cosponsers  # bill index -> cosponsers
        self.bill2subjects_tfidf = bill2subjects_tfidf  # bill index -> subjects
        self.num_nodes = num_nodes  # total number of nodes

    def __len__(self):
        return len(self.bills)

    def __getitem__(self, index):
        bill = self.bills[index]

        cosponsers = self.bill2cosponsers[bill]
        subjects = self.bill2subjects_tfidf[bill]

        results = self.bill2results[bill]

        candidate_pos = results["yeas"]
        candidate_neg = results['nays'] + results['absents']

        if len(candidate_pos) == 0:
            pos_index = self.node2index['pos_legislator']  # legislator padding saying yeas
        else:
            pos_index = random.sample(candidate_pos, 1)[0]
        if len(candidate_neg) == 0:
            neg_index = self.node2index['neg_legislator']  # legislator padding saying nays
        else:
            neg_index = random.sample(candidate_neg, 1)[0]

        return bill, pos_index, neg_index, subjects, cosponsers


class MyMidDataset(Dataset):
    def __init__(self,
                 mids,
                 node2index,
                 mid2results,
                 bill2cosponsers,
                 bill2subjects_tfidf,
                 num_nodes):
        self.mids = mids  # candidate keys
        self.node2index = node2index  # node -> index
        self.mid2results = mid2results  # bill index -> voting results
        self.bill2cosponsers = bill2cosponsers  # bill index -> cosponsers
        self.bill2subjects_tfidf = bill2subjects_tfidf  # bill index -> subjects
        self.num_nodes = num_nodes  # total number of nodes

    def __len__(self):
        return len(self.mids)

    def __getitem__(self, index):
        mid = self.mids[index]
        results = self.mid2results[mid]

        pos_bills = results["yeas"]
        neg_bills = results['nays'] + results['absents']

        if len(pos_bills) == 0:
            pos_bill_index = self.node2index['pos_bill']
            pos_bill_cosponsers = []
            pos_bill_subjects = []
        else:
            pos_bill_index = random.sample(pos_bills, 1)[0]
            pos_bill_cosponsers = self.bill2cosponsers[pos_bill_index]
            pos_bill_subjects = self.bill2subjects_tfidf[pos_bill_index]

        if len(neg_bills) == 0:
            neg_bill_index = self.node2index['neg_bill']
            neg_bill_cosponsers = []
            neg_bill_subjects = []
        else:
            neg_bill_index = random.sample(neg_bills, 1)[0]
            neg_bill_cosponsers = self.bill2cosponsers[neg_bill_index]
            neg_bill_subjects = self.bill2subjects_tfidf[neg_bill_index]

        return mid, \
               pos_bill_index, pos_bill_cosponsers, pos_bill_subjects,\
               neg_bill_index, neg_bill_cosponsers, neg_bill_subjects

def pad_collate(batch):
    max_cosponser_len = float('-inf')
    # max_subject_len = float('-inf')

    for index, line in enumerate(batch):
        subjects, cosponsers = line[3], line[4]
        max_cosponser_len = max(len(cosponsers), max_cosponser_len)
        # max_subject_len = max(len(line[4]), max_subject_len)

    # [1] [1] [1] [Subjects: 30] [Consponsers: Unlimited]
    new_batch = []
    for vid_index, pos_index, neg_index, subjects, cosponsers in batch:
        if len(subjects) == 0:
            subjects = np.array([1])
        if len(cosponsers) == 0:
            cosponsers = np.array([2])
        padded_subjects = np.pad(subjects, (0, 30 - len(subjects)), 'constant', constant_values=0)
        padded_cosponsers = np.pad(cosponsers, (0, max_cosponser_len - len(cosponsers)), 'constant', constant_values=0)
        new_line = [vid_index, pos_index, neg_index] + padded_subjects.tolist() + padded_cosponsers.tolist()
        new_batch.append(new_line)
    return torch.LongTensor(new_batch)


def pad_collate_mids(batch):
    max_pos_cosponser_len = float('-inf')
    max_neg_cosponser_len = float('-inf')
    # [1] [Pos] [Neg] [Pos_Co_L] [Neg_Co_L] [Pos_Subjects: 30] [Neg_Subjects: 30]
    # [Pos_Cosponsers: Unlimited] [Neg_Cosponsers: Unlimited]
    for index, line in enumerate(batch):
        pos_bill_cosponsers, pos_bill_subjects = line[2], line[3]
        neg_bill_cosponsers, neg_bill_subjects = line[5], line[6]
        max_pos_cosponser_len = max(len(pos_bill_cosponsers), max_pos_cosponser_len)
        max_neg_cosponser_len = max(len(neg_bill_cosponsers), max_neg_cosponser_len)

    new_batch = []
    for index, line in enumerate(batch):
        mid, pos_bill_index, neg_bill_index = line[0], line[1], line[4]
        pos_bill_cosponsers, pos_bill_subjects = line[2], line[3]
        neg_bill_cosponsers, neg_bill_subjects = line[5], line[6]

        # TODO: Standardize
        if len(pos_bill_subjects) == 0:
            pos_bill_subjects = np.array([1])
        if len(neg_bill_subjects) == 0:
            neg_bill_subjects = np.array([1])
        if len(pos_bill_cosponsers) == 0:
            pos_bill_cosponsers = np.array([2])
        if len(neg_bill_cosponsers) == 0:
            neg_bill_cosponsers = np.array([2])
        padded_pos_subjects = np.pad(pos_bill_subjects, (0, 30 - len(pos_bill_subjects)), 'constant', constant_values=0)
        padded_neg_subjects = np.pad(neg_bill_subjects, (0, 30 - len(neg_bill_subjects)), 'constant', constant_values=0)

        padded_pos_cosponsers = np.pad(pos_bill_cosponsers, (0, max_pos_cosponser_len - len(pos_bill_cosponsers)),
                                       'constant', constant_values=0)
        padded_neg_cosponsers = np.pad(neg_bill_cosponsers, (0, max_neg_cosponser_len - len(neg_bill_cosponsers)),
                                       'constant', constant_values=0)

        new_line = [mid, pos_bill_index, neg_bill_index, max_pos_cosponser_len, max_neg_cosponser_len] + \
                   padded_pos_subjects.tolist() + padded_neg_subjects.tolist() + \
                   padded_pos_cosponsers.tolist() + padded_neg_cosponsers.tolist()
        new_batch.append(new_line)
    return torch.LongTensor(new_batch)


class MyData(object):
    def __init__(self, load_path):
        self.load_path = load_path
        self.load_data()
        self.num_nodes = len(self.node_list)
        self.build_graph()
        self.num_rels = max(self.edge_type_combined.data).item() + 1

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
        self.node2index = np.load(os.path.join(self.load_path, 'node_index_dict.npy'), allow_pickle=True).item()
        # 所有法案列表 8165
        self.vid_list = np.load(os.path.join(self.load_path, 'vid_list.npy'), allow_pickle=True).tolist()
        # 所有立法者列表 1674
        self.mid_list = np.load(os.path.join(self.load_path, 'mid_list.npy'), allow_pickle=True).tolist()
        # 所有主题列表 1753
        self.subject_list = np.load(os.path.join(self.load_path, 'subject_list.npy'), allow_pickle=True).tolist()
        # 所有委员会列表 44
        # self.committee_list = np.load(os.path.join(self.load_path, 'committee_list.npy'), allow_pickle=True).tolist()
        # 所有党派列表 3 R D I
        # self.party_list = np.load(os.path.join(self.load_path, 'party_list.npy'), allow_pickle=True).tolist()

        # 加载边关系领接矩阵
        self.cosponsor_network_sparse = sp.load_npz(os.path.join(self.load_path, 'cosponsor_network_sparse.npz'))
        self.subject_network_sparse = sp.load_npz(os.path.join(self.load_path, 'subject_network_sparse.npz'))
        self.committee_network_sparse = sp.load_npz(os.path.join(self.load_path, 'committee_network_sparse.npz'))
        self.twitter_network_sparse = sp.load_npz(os.path.join(self.load_path, 'twitter_network_sparse.npz'))
        self.party_network_sparse = sp.load_npz(os.path.join(self.load_path, 'party_network_sparse.npz'))
        self.state_network_sparse = sp.load_npz(os.path.join(self.load_path, 'state_network_sparse.npz'))

        self.committee_network_pairs = self.committee_network_sparse.nonzero()
        self.twitter_network_pairs = self.twitter_network_sparse.nonzero()
        self.party_network_pairs = self.party_network_sparse.nonzero()
        self.state_network_pairs = self.state_network_sparse.nonzero()

        self.bill2cosponsers = matrix2dict(self.cosponsor_network_sparse)
        self.bill2subjects = matrix2dict(self.subject_network_sparse)

        self.cosponsor_network = np.asmatrix(self.cosponsor_network_sparse.toarray())  # 11639, 11639
        self.subject_network = np.asmatrix(self.subject_network_sparse.toarray())
        self.committee_network = np.asmatrix(self.committee_network_sparse.toarray())
        self.twitter_network = np.asmatrix(self.twitter_network_sparse.toarray())
        self.party_network = np.asmatrix(self.party_network_sparse.toarray())
        self.state_network = np.asmatrix(self.state_network_sparse.toarray())

        # 法案-投票记录
        # self.bill2results = np.load(os.path.join(self.load_path, 'vid_results_dict.npy'), allow_pickle=True).item()
        # 法案-投票记录 (实体id替换为 node_list的index)
        self.bill2results = np.load(os.path.join(self.load_path, 'index_results_dict.npy'), allow_pickle=True).item()
        mid_results_dict = np.load(os.path.join(self.load_path, 'mid_results_dict.npy'), allow_pickle=True).item()
        self.mid2results = dict()
        for mid, results in mid_results_dict.items():
            mid = self.node2index[mid]
            this_mid_dict = dict()
            for vote, vote_list in results.items():
                this_mid_dict[vote] = [self.node2index[_] for _ in vote_list]
            self.mid2results[mid] = this_mid_dict

        self.vid_subjects_tfidf_dict = np.load(os.path.join(self.load_path, 'vid_subjects_tfidf_dict.npy'),
                                               allow_pickle=True).item()
        self.bill2subjects_tfidf = dict()

        for vid, subjects_rank in self.vid_subjects_tfidf_dict.items():
            subjects_index_list = [self.node2index[tuple[0]] for tuple in subjects_rank[:30]]
            bill_index = self.node2index[vid]
            self.bill2subjects_tfidf[bill_index] = subjects_index_list

        self.mid_embedding_dict = np.load(os.path.join(self.load_path, 'mid_embedding_dict.npy'),
                                          allow_pickle=True).item()  # 1674
        self.subject_embedding_dict = np.load(os.path.join(self.load_path, 'subject_embedding_dict.npy'),
                                          allow_pickle=True).item()  # 1753
        self.vid_embedding_dict = np.load(os.path.join(self.load_path, 'vid_embedding_dict.npy'),
                                          allow_pickle=True).item()  # 8065

        # 11 + 8165(法案)+1674(立法者)+1753(主题)
        self.null_embeddings = torch.cat([torch.rand(768, 1) for _ in range(11)], dim=1)
        self.mid_embeddings = torch.cat([self.mid_embedding_dict[_].reshape(-1,1) for _ in self.mid_list], dim=1)
        self.subject_embeddings = torch.cat([self.subject_embedding_dict[_].reshape(-1,1) for _ in self.subject_list], dim=1)
        self.vid_embeddings = torch.cat([self.vid_embedding_dict[_].reshape(-1,1) for _ in self.vid_list], dim=1)
        self.pretrained_embeddings = torch.cat([self.null_embeddings, self.vid_embeddings, self.mid_embeddings,
                                                self.subject_embeddings], dim=1).transpose(1,0)

    def build_graph(self):
        edge_index0 = adj_matrix_to_edge_index(self.cosponsor_network)
        edge_index1 = adj_matrix_to_edge_index(self.subject_network)
        edge_index2 = adj_matrix_to_edge_index(self.committee_network)
        edge_index3 = adj_matrix_to_edge_index(self.twitter_network)
        edge_index4 = adj_matrix_to_edge_index(self.party_network)
        edge_index5 = adj_matrix_to_edge_index(self.state_network)

        edge_type0 = torch.zeros(edge_index0.size(1), dtype=torch.long)
        edge_type1 = torch.ones(edge_index1.size(1), dtype=torch.long)
        edge_type2 = 2 * torch.ones(edge_index2.size(1), dtype=torch.long)
        edge_type3 = 3 * torch.ones(edge_index3.size(1), dtype=torch.long)
        edge_type4 = 4 * torch.ones(edge_index4.size(1), dtype=torch.long)
        edge_type5 = 5 * torch.ones(edge_index5.size(1), dtype=torch.long)

        self.edge_index_combined = torch.cat([edge_index0, edge_index1, edge_index2,
                                              edge_index3, edge_index4, edge_index5], dim=1)
        self.edge_type_combined = torch.cat([edge_type0, edge_type1, edge_type2,
                                             edge_type3, edge_type4, edge_type5])

        graph = dgl.graph((self.edge_index_combined[0], self.edge_index_combined[1]), num_nodes=self.num_nodes)
        graph.edata['etype'] = self.edge_type_combined
        self.graph = graph

    def get_dataset_vids(self, cidstart):
        # train data
        window = self.cid_window_dict[cidstart]
        self.train_vids = []
        for cid in window[:4]:
            vids = self.cid_vids_dict[cid]
            self.train_vids += vids
        self.train_vids = [self.node2index[_] for _ in self.train_vids]

        # val / test data
        cid = cidstart + 4
        vids = self.cid_vids_dict[cid]
        random.shuffle(vids)
        val_size = len(vids) // 2
        self.val_vids, self.test_vids = vids[:val_size], vids[val_size:]
        self.val_vids = [self.node2index[_] for _ in self.val_vids]
        self.test_vids = [self.node2index[_] for _ in self.test_vids]

    def get_dataset_mids(self, cidstart):
        # train data
        window = self.cid_window_dict[cidstart]
        self.train_mids = []
        for cid in window[:4]:
            mids = self.cid_mids_dict[cid]
            self.train_mids += mids
        self.train_mids = [self.node2index[_] for _ in self.train_mids]

        # val / test data
        cid = cidstart + 4
        mids = self.cid_mids_dict[cid]
        random.shuffle(mids)
        val_size = len(mids) // 2
        self.val_mids, self.test_mids = mids[:val_size], mids[val_size:]
        self.val_mids = [self.node2index[_] for _ in self.val_mids]
        self.test_mids = [self.node2index[_] for _ in self.test_mids]

    def get_train_dataset_mids(self):
        return MyMidDataset(
            mids=self.train_mids,
            node2index=self.node2index,
            mid2results=self.mid2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf,
            num_nodes=self.num_nodes
        )

    def get_val_dataset_mids(self):
        return MyMidDataset(
            mids=self.val_mids,
            node2index=self.node2index,
            mid2results=self.mid2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf,
            num_nodes=self.num_nodes
        )

    def get_test_dataset_mids(self):
        return MyMidDataset(
            mids=self.test_mids,
            node2index=self.node2index,
            mid2results=self.mid2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf,
            num_nodes=self.num_nodes
        )

    def get_train_dataset(self):
        return MyDataset(
            bills=self.train_vids,
            node2index=self.node2index,
            bill2results=self.bill2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf,
            num_nodes=self.num_nodes
        )

    def get_val_dataset(self):
        return MyDataset(
            bills=self.val_vids,
            node2index=self.node2index,
            bill2results=self.bill2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf,
            num_nodes=self.num_nodes
        )

    def get_test_dataset(self):
        return MyDataset(
            bills=self.test_vids,
            node2index=self.node2index,
            bill2results=self.bill2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf,
            num_nodes=self.num_nodes
        )


