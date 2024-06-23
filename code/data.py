import os
import random
from collections import defaultdict

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from loguru import logger
from torch.utils.data import Dataset

from utils import adj_matrix_to_edge_index, matrix2dict


class MyMidDataset(Dataset):
    def __init__(self,
                 mids,
                 node2index,
                 mid2results,
                 bill2cosponsers,
                 bill2subjects_tfidf):
        self.mids = mids  # candidate
        self.node2index = node2index  # node -> index
        self.mid2results = mid2results  # bill index -> voting results
        self.bill2cosponsers = bill2cosponsers  # bill index -> cosponsers
        self.bill2subjects_tfidf = bill2subjects_tfidf  # bill index -> subjects

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
               pos_bill_index, pos_bill_cosponsers, pos_bill_subjects, \
               neg_bill_index, neg_bill_cosponsers, neg_bill_subjects


def pad_collate_mids(batch):
    max_pos_cosponser_len = 1  # float('-inf')
    max_neg_cosponser_len = 1  # float('-inf')
    # [mid] [Pos] [Neg] [Pos_Co_L] [Neg_Co_L] [Pos_Subjects: 30] [Neg_Subjects: 30]
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
    def __init__(self,
                 data_path,
                 if_cold_start):
        self.data_path = data_path
        self.if_cold_start = if_cold_start
        self.load_data()

        self.num_nodes = len(self.node2index)
        self.num_rels = 7

    def load_data(self):
        # 加载实体   法案立法者与国会届次相关, 主题委员会党派与国会届次无关
        # 国会届次 102-116
        self.cid_list = np.load(os.path.join(self.data_path, 'cid_list.npy'), allow_pickle=True).tolist()
        # 时间窗口起始点 102-112
        self.cidstart_list = np.load(os.path.join(self.data_path, 'cidstart_list.npy'), allow_pickle=True).tolist()
        # 国会届次-时间窗口
        self.cid_window_dict = np.load(os.path.join(self.data_path, 'cidstart_window_dict.npy'),
                                       allow_pickle=True).item()
        # 国会届次-法案(众)
        self.cid_vids_dict = np.load(os.path.join(self.data_path, 'cid_vids_dict.npy'), allow_pickle=True).item()
        # 国会届次-立法者(参+众)
        self.cid_mids_dict = np.load(os.path.join(self.data_path, 'cid_mids_dict.npy'), allow_pickle=True).item()
        # 国会届次-立法者(众)
        self.cid_hmids_dict = np.load(os.path.join(self.data_path, 'cid_hmids_dict.npy'), allow_pickle=True).item()
        # 国会届次-立法者(参)
        # self.cid_smids_dict = np.load(os.path.join(self.load_path, 'cid_smids_dict.npy'), allow_pickle=True).item()

        # 所有实体列表 11639 = 8165(法案)+1674(立法者)+1753(主题)+44(委员会)+3(党派)
        self.node_list = np.load(os.path.join(self.data_path, 'node_list.npy'), allow_pickle=True).tolist()
        # 所有实体index
        self.node2index = np.load(os.path.join(self.data_path, 'node_index_dict.npy'), allow_pickle=True).item()
        # 所有法案列表 8165
        self.vid_list = np.load(os.path.join(self.data_path, 'vid_list.npy'), allow_pickle=True).tolist()
        # 所有立法者列表 1674
        self.mid_list = np.load(os.path.join(self.data_path, 'mid_list.npy'), allow_pickle=True).tolist()
        # 所有主题列表 1753
        self.subject_list = np.load(os.path.join(self.data_path, 'subject_list.npy'), allow_pickle=True).tolist()
        # 所有委员会列表 44
        # self.committee_list = np.load(os.path.join(self.load_path, 'committee_list.npy'), allow_pickle=True).tolist()
        # 所有党派列表 3 R D I
        # self.party_list = np.load(os.path.join(self.load_path, 'party_list.npy'), allow_pickle=True).tolist()

        # heterogeneous network
        self.cosponsor_network_sparse = sp.load_npz(os.path.join(self.data_path, 'cosponsor_network_sparse.npz'))
        self.subject_network_sparse = sp.load_npz(os.path.join(self.data_path, 'subject_network_sparse.npz'))
        self.committee_network_sparse = sp.load_npz(os.path.join(self.data_path, 'committee_network_sparse.npz'))
        self.twitter_network_sparse = sp.load_npz(os.path.join(self.data_path, 'twitter_network_sparse.npz'))
        self.party_network_sparse = sp.load_npz(os.path.join(self.data_path, 'party_network_sparse.npz'))
        self.state_network_sparse = sp.load_npz(os.path.join(self.data_path, 'state_network_sparse.npz'))

        self.committee_network_pairs = self.committee_network_sparse.nonzero()
        self.twitter_network_pairs = self.twitter_network_sparse.nonzero()
        self.party_network_pairs = self.party_network_sparse.nonzero()
        self.state_network_pairs = self.state_network_sparse.nonzero()

        self.bill2cosponsers = matrix2dict(self.cosponsor_network_sparse)  # (array[0], array[1])
        self.bill2subjects = matrix2dict(self.subject_network_sparse)

        # self.cosponsor_network = np.asmatrix(self.cosponsor_network_sparse.toarray())  # 11639, 11639
        # self.subject_network = np.asmatrix(self.subject_network_sparse.toarray())
        self.committee_network = np.asmatrix(self.committee_network_sparse.toarray())
        self.twitter_network = np.asmatrix(self.twitter_network_sparse.toarray())
        self.party_network = np.asmatrix(self.party_network_sparse.toarray())
        self.state_network = np.asmatrix(self.state_network_sparse.toarray())

        # bill - vote - vote_list (legislators)
        self.bill2results = np.load(os.path.join(self.data_path, 'index_results_dict.npy'), allow_pickle=True).item()
        mid_results_dict = np.load(os.path.join(self.data_path, 'mid_results_dict.npy'), allow_pickle=True).item()
        self.mid2results = dict()  # legislator - vote - vote_list (bills)
        for mid, results in mid_results_dict.items():
            mid = self.node2index[mid]
            this_mid_dict = dict()
            for vote, vote_list in results.items():
                this_mid_dict[vote] = [self.node2index[_] for _ in vote_list]
            self.mid2results[mid] = this_mid_dict
        # legislator - congress - vote - vote_list (bills)
        mid_cid_results_dict = np.load(os.path.join(self.data_path, 'mid_cid_results_dict.npy'),
                                       allow_pickle=True).item()

        self.mid_cid2results = dict()
        for mid, cid_results in mid_cid_results_dict.items():
            mid = self.node2index[mid]
            this_mid_cid_dict = dict()
            for cid, results in cid_results.items():
                this_cid_dict = dict()
                for vote, vote_list in results.items():
                    this_cid_dict[vote] = [self.node2index[_] for _ in vote_list]
                this_mid_cid_dict[cid] = this_cid_dict
            self.mid_cid2results[mid] = this_mid_cid_dict
        # bill - subjects (descend by TF-IDF values)
        self.vid_subjects_tfidf_dict = np.load(os.path.join(self.data_path, 'vid_subjects_tfidf_dict.npy'),
                                               allow_pickle=True).item()
        self.bill2subjects_tfidf = dict()

        for vid, subjects_rank in self.vid_subjects_tfidf_dict.items():
            subjects_index_list = [self.node2index[tuple[0]] for tuple in subjects_rank[:30]]
            bill_index = self.node2index[vid]
            self.bill2subjects_tfidf[bill_index] = subjects_index_list

        self.mid_embedding_dict = np.load(os.path.join(self.data_path, 'mid_embedding_dict.npy'),
                                          allow_pickle=True).item()  # 1674
        self.subject_embedding_dict = np.load(os.path.join(self.data_path, 'subject_embedding_dict.npy'),
                                              allow_pickle=True).item()  # 1753
        self.vid_embedding_dict = np.load(os.path.join(self.data_path, 'vid_embedding_dict.npy'),
                                          allow_pickle=True).item()

    def load_pretrained_embeddings(self):
        self.null_embeddings = torch.cat([torch.rand(768, 1) for _ in range(11)], dim=1)  # 11 padding
        self.mid_embeddings = torch.cat([self.mid_embedding_dict[_].reshape(-1, 1) for _ in self.mid_list],
                                        dim=1)  # bills
        self.subject_embeddings = torch.cat([self.subject_embedding_dict[_].reshape(-1, 1) for _ in self.subject_list],
                                            dim=1)  # legislators
        self.vid_embeddings = torch.cat([self.vid_embedding_dict[_].reshape(-1, 1) for _ in self.vid_list],
                                        dim=1)  # subjects
        self.pretrained_embeddings = torch.cat([self.null_embeddings, self.vid_embeddings, self.mid_embeddings,
                                                self.subject_embeddings], dim=1).transpose(1, 0)
        return self.pretrained_embeddings

    def update_historical_interactions(self, cid_latest):
        # legislator - bill interactions
        valid_range = range(self.cid_list[0], cid_latest)  # fatal error
        lb_rows, lb_cols = [], []
        bs_rows, bs_cols = [], []
        for mid, cid_results in self.mid_cid2results.items():
            for cid, results in cid_results.items():
                if cid in valid_range:
                    valid_bills = list(set(results['proposals']))  # results['yeas'] +
                    for bill in valid_bills:
                        lb_rows.append(mid)
                        lb_cols.append(bill)

        # bill - subject interactions
        for cid, bills in self.cid_vids_dict.items():  # differ: bill2subjects
            if cid in valid_range:
                for bill in bills:
                    bill = self.node2index[bill]
                    subjects = self.bill2subjects[bill]
                    for subject in subjects:
                        bs_rows.append(bill)
                        bs_cols.append(subject)

        # to bidirectional
        lb_pairs = (lb_rows + lb_cols, lb_cols + lb_rows)
        bs_pairs = (bs_rows + bs_cols, bs_cols + bs_rows)
        return lb_pairs, bs_pairs

    def build_graph(self, cid_latest):
        lb_pairs, bs_pairs = self.update_historical_interactions(cid_latest)  # dynamically update
        lb_edge_indexes = torch.stack([torch.LongTensor(lb_pairs[0]), torch.LongTensor(lb_pairs[1])], dim=0)
        bs_edge_indexes = torch.stack([torch.LongTensor(bs_pairs[0]), torch.LongTensor(bs_pairs[1])], dim=0)

        committee_edge_indexes = adj_matrix_to_edge_index(self.committee_network)
        twitter_edge_indexes = adj_matrix_to_edge_index(self.twitter_network)
        party_edge_indexes = adj_matrix_to_edge_index(self.party_network)
        state_edge_indexes = adj_matrix_to_edge_index(self.state_network)

        lb_edge_types = torch.zeros(lb_edge_indexes.size(1), dtype=torch.long)
        bs_edge_types = torch.ones(bs_edge_indexes.size(1), dtype=torch.long)
        committee_edge_types = 2 * torch.ones(committee_edge_indexes.size(1), dtype=torch.long)
        twitter_edge_types = 3 * torch.ones(twitter_edge_indexes.size(1), dtype=torch.long)
        party_edge_types = 4 * torch.ones(party_edge_indexes.size(1), dtype=torch.long)
        state_edge_type = 5 * torch.ones(state_edge_indexes.size(1), dtype=torch.long)

        self.edge_indexes = torch.cat([lb_edge_indexes, bs_edge_indexes, committee_edge_indexes,
                                       twitter_edge_indexes, party_edge_indexes, state_edge_indexes], dim=1)
        self.edge_types = torch.cat([lb_edge_types, bs_edge_types, committee_edge_types,
                                     twitter_edge_types, party_edge_types, state_edge_type])

        graph = dgl.graph((self.edge_indexes[0], self.edge_indexes[1]), num_nodes=self.num_nodes)
        graph.edata['etype'] = self.edge_types

        logger.info('[Data] cid_latest={}, edges={}'.format(cid_latest, len(self.edge_indexes[0])))
        return graph

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
        self.window = self.cid_window_dict[cidstart]
        self.train_cids = self.window[:4]
        self.val_cids = self.test_cids = [self.window[-1]]
        self.train_mids = []
        for cid in self.train_cids:
            mids = self.cid_mids_dict[cid]
            self.train_mids += mids
        self.train_mids = list(set(self.train_mids))
        self.train_mids.sort()
        random.shuffle(self.train_mids)
        self.train_mids = [self.node2index[_] for _ in self.train_mids]

        # val / test data
        cid = cidstart + 4
        full_mids = [self.node2index[_] for _ in self.cid_mids_dict[cid]]
        mids = list(set(self.train_mids) & set(full_mids)) if self.if_cold_start else full_mids
        mids.sort()
        random.shuffle(mids)
        val_size = len(mids) // 2
        self.val_mids, self.test_mids = mids[:val_size], mids[val_size:]

        self.train_mid2results = dict()
        for mid in self.train_mids:
            this_mid_dict = defaultdict(list)
            for cid in self.train_cids:
                for vote, vote_list in self.mid_cid2results[mid][cid].items():
                    this_mid_dict[vote] += vote_list
            self.train_mid2results[mid] = this_mid_dict

        self.val_mid2results = dict()
        for mid in self.val_mids:
            this_mid_dict = defaultdict(list)
            for cid in self.val_cids:
                for vote, vote_list in self.mid_cid2results[mid][cid].items():
                    this_mid_dict[vote] += vote_list
            self.val_mid2results[mid] = this_mid_dict

        self.test_mid2results = dict()
        for mid in self.test_mids:
            this_mid_dict = defaultdict(list)
            for cid in self.test_cids:
                for vote, vote_list in self.mid_cid2results[mid][cid].items():
                    this_mid_dict[vote] += vote_list
            self.test_mid2results[mid] = this_mid_dict

        logger.info('[Data] cid={}, train_mids={}, full_mids={}, valid_mids={}, val_mids={}, test_mids={}'.format(
            cidstart, len(self.train_mids), len(full_mids), len(mids), len(self.val_mids), len(self.test_mids)))

    def get_train_dataset_mids(self):
        return MyMidDataset(
            mids=self.train_mids,
            node2index=self.node2index,
            mid2results=self.train_mid2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf
        )

    def get_val_dataset_mids(self):
        return MyMidDataset(
            mids=self.val_mids,
            node2index=self.node2index,
            mid2results=self.val_mid2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf
        )

    def get_test_dataset_mids(self):
        return MyMidDataset(
            mids=self.test_mids,
            node2index=self.node2index,
            mid2results=self.test_mid2results,
            bill2cosponsers=self.bill2cosponsers,
            bill2subjects_tfidf=self.bill2subjects_tfidf
        )
