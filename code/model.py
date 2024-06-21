import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from hgb import myGAT


class MergeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=dim, out_features=dim // 2)
        self.fc2 = nn.Linear(in_features=dim // 2, out_features=1)
        self.act = nn.ReLU()

    def initialize(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = x.float()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class FFNN(nn.Module):
    def __init__(self, input_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim // 2)
        self.fc2 = nn.Linear(in_features=input_dim // 2, out_features=1)
        self.act = nn.ReLU()

    def initialize(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, bill_embeddings, legistor_embeddings):
        x = torch.cat([bill_embeddings, legistor_embeddings], dim=1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

    def forward(self, bill_embeddings, legislator_embeddings):
        scores = (bill_embeddings * legislator_embeddings).sum(axis=1)
        return scores


class RGCNModel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 edge_index,
                 edge_type):
        super(RGCNModel, self).__init__()
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.conv1 = RGCNConv(in_channels=in_channels, out_channels=out_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(in_channels=out_channels, out_channels=out_channels, num_relations=num_relations)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x=x, edge_index=self.edge_index, edge_type=self.edge_type)
        x = self.act(x)
        x = self.conv2(x=x, edge_index=self.edge_index, edge_type=self.edge_type)
        return x


class DualAttention(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 dropout):
        super(DualAttention, self).__init__()
        self.dropout = dropout
        self.n_head = num_heads
        self.d_model = d_model

        self.left_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head, dropout=self.dropout)
        self.right_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head, dropout=self.dropout)

        # self.bill2cosponsers = data.bill2cosponsers
        # self.bill2subjects = data.bill2subjects

    def forward(self, node_embeddings, query_idx, sponser_idx, subject_idx, sponser_masks, subject_masks):
        query_embeddings = node_embeddings[query_idx].unsqueeze(0)  # (1, bsz, dim)
        sponser_embeddings = node_embeddings[sponser_idx].transpose(0, 1)  # (num_sponsers, bsz, dim)
        subject_embeddings = node_embeddings[subject_idx].transpose(0, 1)  # (num_subjects, bsz, dim)

        # sponser_masks = sponser_masks.transpose(0, 1)
        # subject_masks = subject_masks.transpose(0, 1)

        left_embeddings, left_weights = self.left_attn(query=query_embeddings,
                                                       key=sponser_embeddings,
                                                       value=sponser_embeddings,
                                                       key_padding_mask=sponser_masks,  # (bsz, num_sponsers)
                                                       need_weights=True)
        # (1, bsz, dim), (bsz, 1, num_sponsers)

        right_embeddings, right_weights = self.right_attn(query=query_embeddings,
                                                          key=subject_embeddings,
                                                          value=subject_embeddings,
                                                          key_padding_mask=subject_masks,
                                                          need_weights=True)

        left_embeddings = left_embeddings.squeeze(0)  # (bsz, dim)
        right_embeddings = right_embeddings.squeeze(0)  # (bsz, dim)

        left_weights = left_weights.squeeze(1)
        right_weight = right_weights.squeeze(1)

        return left_embeddings, right_embeddings


class RGCN_DualAttn_FFNN(nn.Module):
    def __init__(self,
                 dim,
                 num_nodes,
                 num_relations,
                 num_layers,
                 num_heads,
                 dropout_1,
                 dropout_2,
                 negative_slope,
                 lambda_1,
                 lambda_2,
                 alpha,
                 pretrained,
                 data):
        super(RGCN_DualAttn_FFNN, self).__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.n_head = num_heads
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.negative_slope = negative_slope
        self.data = data
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha

        self.pretrained = pretrained

        if self.pretrained is not None:
            self.node_embeddings = nn.Parameter(torch.FloatTensor(self.num_nodes, 768))
            self.LinearLayer = nn.Linear(768, self.dim)
            nn.init.normal_(self.LinearLayer.weight, mean=0, std=0.01)  # 正态分布初始化权重
            nn.init.constant_(self.LinearLayer.bias, 0)  # 设置偏置为0
        else:
            self.node_embeddings = nn.Parameter(torch.FloatTensor(self.num_nodes, self.dim))

        self.pre_encoder = 'HGB'
        if self.pre_encoder == 'HGB':
            self.HGB = myGAT(
                edge_dim=self.dim,
                num_etypes=self.num_relations,
                in_dims=[self.dim],
                num_hidden=self.dim,
                num_layers=self.num_layers,
                heads=[self.n_head] * self.num_layers,
                activation=F.relu,
                feat_drop=self.dropout_1,
                attn_drop=self.dropout_1,
                negative_slope=self.negative_slope,
                residual=False,
                alpha=self.alpha)

        elif self.pre_encoder == 'RGCN':
            self.RGCN = RGCNModel(
                in_channels=self.dim,
                out_channels=self.dim,
                num_relations=self.num_relations,
                edge_index=self.edge_index,
                edge_type=self.edge_type)

        self.DualAttn = DualAttention(d_model=self.dim,
                                      num_heads=self.n_head,
                                      dropout=self.dropout_2)

        self.FusionAttn = nn.MultiheadAttention(embed_dim=self.dim * 2,
                                                num_heads=self.n_head
                                                )
        self.LegislatorMLP = nn.Linear(in_features=self.dim * 2, out_features=self.dim)
        self.PredictorLayer = Predictor()

        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.LegislatorMLP.weight)
        if self.pretrained is not None:
            self.node_embeddings.data.copy_(self.pretrained)
        else:
            nn.init.xavier_normal_(self.node_embeddings)

    def forward(self, batch, graph):
        # TODO: Check inputs
        mid_batch, pos_bill_index_batch, neg_bill_index_batch, max_pos_cosponser_len_batch, max_neg_cosponser_len_batch \
            = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
        pos_subject_batch, neg_subject_batch = batch[:, 5: 35], batch[:, 35: 65]
        pos_subject_masks = pos_subject_batch == 0
        neg_subject_masks = neg_subject_batch == 0

        max_pos_cosponser_len = max_pos_cosponser_len_batch[0]
        max_neg_cosponser_len = max_neg_cosponser_len_batch[0]

        pos_cosponser_batch, neg_cosponser_batch = batch[:, 65: 65 + max_pos_cosponser_len], \
                                                   batch[:, 65 + max_pos_cosponser_len:]
        pos_cosponser_masks, neg_cosponser_masks = pos_cosponser_batch == 0, neg_cosponser_batch == 0

        if self.pretrained is not None:
            x = self.LinearLayer(self.node_embeddings)
        else:
            x = self.node_embeddings

        if self.pre_encoder == 'RGCN':
            node_embeddings = self.RGCN(x=x)  # (num_nodes, dim)
        elif self.pre_encoder == 'HGB':
            node_embeddings = self.HGB(features_list=[x],
                                       e_feat=graph.edata['etype'],
                                       g=graph
                                       )
        else:
            raise NotImplementedError

        legis_embeddings = node_embeddings[mid_batch]  # (bsz, dim)
        # pos_bill_embeddings = node_embeddings[pos_bill_index_batch]
        # neg_bill_embeddings = node_embeddings[neg_bill_index_batch]

        pos_left_embeddings, pos_right_embeddings = self.DualAttn(node_embeddings=node_embeddings,
                                                                  query_idx=mid_batch,
                                                                  sponser_idx=pos_cosponser_batch,
                                                                  subject_idx=pos_subject_batch,
                                                                  sponser_masks=pos_cosponser_masks,
                                                                  subject_masks=pos_subject_masks)  # (bsz, dim)

        pos_embeddings = torch.cat([pos_left_embeddings, pos_right_embeddings], dim=1)  # (bsz, 2 * dim)
        pos_embeddings = pos_embeddings.unsqueeze(0)  # (1, bsz, 2*dim)

        pos_embeddings, _ = self.FusionAttn(query=pos_embeddings,
                                            key=pos_embeddings,
                                            value=pos_embeddings)
        pos_embeddings = pos_embeddings.squeeze(0)  # (bsz, 2*dim)
        pos_embeddings = self.LegislatorMLP(pos_embeddings)  # (bsz, dim)

        neg_left_embeddings, neg_right_embeddings = self.DualAttn(node_embeddings=node_embeddings,
                                                                  query_idx=mid_batch,
                                                                  sponser_idx=neg_cosponser_batch,
                                                                  subject_idx=neg_subject_batch,
                                                                  sponser_masks=neg_cosponser_masks,
                                                                  subject_masks=neg_subject_masks)

        neg_embeddings = torch.cat([neg_left_embeddings, neg_right_embeddings], dim=1)
        neg_embeddings = neg_embeddings.unsqueeze(0)
        neg_embeddings, _ = self.FusionAttn(query=neg_embeddings,
                                            key=neg_embeddings,
                                            value=neg_embeddings)
        neg_embeddings = neg_embeddings.squeeze(0)
        neg_embeddings = self.LegislatorMLP(neg_embeddings)

        # pos_scores = self.FFNN(left_embeddings=pos_left_embeddings,
        #                        right_embeddings=pos_right_embeddings,
        #                        legistor_embeddings=pos_bill_embeddings)
        #
        # neg_scores = self.FFNN(left_embeddings=neg_left_embeddings,
        #                        right_embeddings=neg_right_embeddings,
        #                        legistor_embeddings=neg_bill_embeddings)

        pos_scores = self.PredictorLayer(bill_embeddings=pos_embeddings,
                                         legislator_embeddings=legis_embeddings)  # (bsz)

        neg_scores = self.PredictorLayer(bill_embeddings=neg_embeddings,
                                         legislator_embeddings=legis_embeddings)  # (bsz)

        # pos_scores = pos_scores.squeeze(-1)
        # neg_scores = neg_scores.squeeze(-1)

        return self.cal_loss(pos_scores, neg_scores, node_embeddings)

    # def forward(self, batch):
    #     # TODO: Check inputs
    #     # bsz, data_len = batch.shape
    #     vid_index_batch, pos_index_batch, neg_index_batch = batch[:, 0], batch[:, 1], batch[:, 2]
    #     subject_batch, cosponser_batch = batch[:, 3:33], batch[:, 33:]
    #     subject_masks, cosponser_masks = subject_batch == 0, cosponser_batch == 0
    #     if self.pretrained is not None:
    #         x = self.LinearLayer(self.node_embeddings)
    #     else:
    #         x = self.node_embeddings
    #
    #     if self.pre_encoder == 'RGCN':
    #         node_embeddings = self.RGCN(x=x)  # (num_nodes, dim)
    #     elif self.pre_encoder == 'HGB':
    #         node_embeddings = self.HGB(features_list=[x],
    #                                    e_feat=self.graph.edata['etype'],
    #                                    )
    #     else:
    #         raise NotImplementedError
    #
    #     pos_legis_embeddings = node_embeddings[pos_index_batch]
    #     neg_legis_embeddings = node_embeddings[neg_index_batch]
    #
    #     left_embeddings, right_embeddings = self.DualAttn(node_embeddings=node_embeddings,
    #                                                       bill_idx=vid_index_batch,
    #                                                       sponser_idx=cosponser_batch,
    #                                                       subject_idx=subject_batch,
    #                                                       sponser_masks=cosponser_masks,
    #                                                       subject_masks=subject_masks)
    #
    #     pos_scores = self.FFNN(left_embeddings=left_embeddings,
    #                            right_embeddings=right_embeddings,
    #                            legistor_embeddings=pos_legis_embeddings)
    #
    #     neg_scores = self.FFNN(left_embeddings=left_embeddings,
    #                            right_embeddings=right_embeddings,
    #                            legistor_embeddings=neg_legis_embeddings)
    #
    #     pos_scores = pos_scores.squeeze(-1)
    #     neg_scores = neg_scores.squeeze(-1)
    #
    #     return self.cal_loss(pos_scores, neg_scores, node_embeddings)

    def cal_sim_loss(self, pairs, batch_size, node_embeddings):
        indices = np.random.permutation(len(pairs[0]))
        valid_indices = list(set(list(pairs[0]) + list(pairs[1])))
        sample_indices = list(indices[:batch_size])

        fro = list(pairs[0][sample_indices])
        to = list(pairs[1][sample_indices])
        another = [np.random.choice(valid_indices) for _ in range(batch_size)]

        fro_embeddings = node_embeddings[torch.LongTensor(fro)]
        to_embeddings = node_embeddings[torch.LongTensor(to)]
        another_embeddings = node_embeddings[torch.LongTensor(another)]

        pos_scores = self.PredictorLayer(fro_embeddings, to_embeddings)
        neg_scores = self.PredictorLayer(fro_embeddings, another_embeddings)

        loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores)))

        # for i in range(len(fro)):
        #     fro_emb, to_emb = node_embeddings[fro[i]], node_embeddings[to[i]]
        #     # loss += -torch.log(torch.sigmoid(torch.dot(fro_emb, to_emb)))
        #     for j in range(self.negative_sample):
        #         another = np.random.choice(valid_indices)
        #         another_emb = node_embeddings[another]
        #         loss += -torch.log(torch.sigmoid(torch.dot(fro_emb, to_emb) - torch.dot(fro_emb, another_emb)))
        # loss = loss / batch_size
        return loss

    def cal_priority_loss(self, batch_size, node_embeddings):
        legislators, pos_bills, neg_bills = [], [], []
        while True:
            legislator = np.random.choice(self.data.train_mids)
            candidate_pos_bills = self.data.train_mid2results[legislator]['proposals']
            candidate_neg_bills = self.data.train_mid2results[legislator]['yeas']

            if len(candidate_pos_bills) == 0:
                continue
            else:
                pos_bill = np.random.choice(candidate_pos_bills)

            if len(candidate_neg_bills) == 0:
                continue
            else:
                neg_bill = np.random.choice(candidate_neg_bills)

            legislators.append(legislator)
            pos_bills.append(pos_bill)
            neg_bills.append(neg_bill)

            if len(legislators) >= batch_size:
                break

        fro_embeddings = node_embeddings[torch.LongTensor(legislators)]
        to_embeddings = node_embeddings[torch.LongTensor(pos_bills)]
        another_embeddings = node_embeddings[torch.LongTensor(neg_bills)]

        pos_scores = self.PredictorLayer(fro_embeddings, to_embeddings)
        neg_scores = self.PredictorLayer(fro_embeddings, another_embeddings)

        loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores)))

        return loss

    def cal_loss(self, pos_pre, neg_pre, node_embeddings):
        batch_size = len(pos_pre)
        # loss roll call prediction
        targets = torch.cat([torch.ones_like(pos_pre), torch.zeros_like(pos_pre)], dim=0)
        predictions = torch.cat([pos_pre, neg_pre], dim=0)
        main_loss = F.binary_cross_entropy_with_logits(predictions, targets)

        # loss_1 group similarity
        loss_1 = 0
        # committee / twitter / state / party
        if self.lambda_1 > 0:
            loss_1 += self.cal_sim_loss(self.data.committee_network_pairs, batch_size, node_embeddings)
            loss_1 += self.cal_sim_loss(self.data.state_network_pairs, batch_size, node_embeddings)
            loss_1 += self.cal_sim_loss(self.data.twitter_network_pairs, batch_size, node_embeddings)
            loss_1 += self.cal_sim_loss(self.data.party_network_pairs, batch_size, node_embeddings)
            loss_1 = loss_1 / 4

        # loss_2 sponsorship priority
        loss_2 = 0
        if self.lambda_2 > 0:
            loss_2 += self.cal_priority_loss(batch_size, node_embeddings)

        loss = main_loss + self.lambda_1 * loss_1 + self.lambda_2 * loss_2

        # loss = loss / batch_size
        predictions = torch.sigmoid(predictions)
        predicted_labels = torch.round(predictions)
        targets = targets.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        predicted_labels = predicted_labels.detach().cpu().numpy()

        return loss, targets, predictions, predicted_labels

#
# class RGCN_Merge(nn.Module):
#     def __init__(self, num_nodes, input_size, hidden_size, output_size, num_relations):
#         super(RGCN_Merge, self).__init__()
#         self.num_nodes = num_nodes  # all nodes
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_relations = num_relations
#
#         self.node_embeddings = nn.Parameter(torch.FloatTensor(self.num_nodes, self.input_size))
#
#         self.HGNN = RGCNModel(self.input_size, 64, self.num_relations)
#         self.affinity_score = MergeLayer(128)
#
#         self.initialize()
#
#     def initialize(self):
#         nn.init.xavier_normal_(self.node_embeddings)
#
#     def forward(self, batch, device, edge_index_combined, edge_type_combined):
#         _, info = batch
#         bill_id, user1_id, user2_id = info
#         bill_id = bill_id.to(device)
#         user1_id = user1_id.to(device)
#         user2_id = user2_id.to(device)
#
#         edge_index_combined = edge_index_combined.to(device)
#         edge_type_combined = edge_type_combined.to(device)
#         nodes_embeddings = self.HGNN(self.users_feature, edge_index_combined, edge_type_combined).to(device)
#
#         user1_embedding = nodes_embeddings[user1_id]  # pos
#         user2_embedding = nodes_embeddings[user2_id]  # neg
#         bill_embedding = nodes_embeddings[bill_id]
#
#         pos_pre = self.affinity_score(bill_embedding, user1_embedding)
#         neg_pre = self.affinity_score(bill_embedding, user2_embedding)
#
#         bce_loss, accuracy, f1, recall, precision, auc = cal_loss(pos_pre, neg_pre)
#
#         return bce_loss, accuracy, f1, recall, precision, auc

# class HyperConv(nn.Module):
#     def __init__(self, in_channels, out_channels, G, bias, residual):
#         super(HyperConv, self).__init__()
#         self.G = G
#         self.residual = residual
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, x):
#         x_self = x
#         x = torch.matmul(x, self.weight)
#         if self.bias is not None:
#             x = x + self.bias
#         x_conv = torch.matmul(self.G, x)
#         if self.residual:
#             x_conv = x_conv + x_self
#         return F.relu(x_conv)
#
#
# class HyperGNN(nn.Module):
#     def __init__(self, in_channel, out_channel, hid_channel, G, bias=False, residual=False):
#         super(HyperGNN, self).__init__()
#         self.layer_size = [in_channel] + hid_channel + [out_channel]
#         self.layers = nn.Sequential()
#
#         for k in range(len(self.layer_size) - 1):
#             self.layers.add_module('hc_%d' % k,
#                                    HyperConv(self.layer_size[k], self.layer_size[k+1], G, bias, residual))
#             if k != len(self.layer_size) - 2:
#                 self.layers.add_module('relu_%d' % k, nn.ReLU())
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# class HyperGNN_Merge(nn.Module):
#     def __init__(self, embedding_size, hidden_size, output_size, G):
#         super(HyperGNN_Merge, self).__init__()
#
#         self.HyperGNN = HyperGNN(in_channel= embedding_size,
#                                  out_channel=output_size,
#                                  hid_channel=hidden_size,
#                                  G=G)
#         self.affinity_score = MergeLayer(128)
#         self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
#
#         self.init_emb()
#
#     def init_emb(self):
#         self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
#
#     def forward(self, batch, device, edge_index_combined, edge_type_combined):
#         _, info = batch
#         bill_id, user1_id, user2_id = info
#         bill_id = bill_id.to(device)
#         user1_id = user1_id.to(device)
#         user2_id = user2_id.to(device)
#
#         # edge_index_combined = edge_index_combined.to(device)
#         # edge_type_combined = edge_type_combined.to(device)
#         # nodes_embeddings = self.RGCNModel(self.users_feature, edge_index_combined, edge_type_combined).to(device)
#         nodes_embeddings = self.HyperGNN(self.users_feature)
#
#         user1_embedding = nodes_embeddings[user1_id]  # pos
#         user2_embedding = nodes_embeddings[user2_id]  # neg
#         bill_embedding = nodes_embeddings[bill_id]
#
#         pos_pre = self.affinity_score(bill_embedding, user1_embedding)
#         neg_pre = self.affinity_score(bill_embedding, user2_embedding)
#
#         bce_loss, accuracy, f1, recall, precision, auc = cal_loss(pos_pre, neg_pre)
#
#         return bce_loss, accuracy, f1, recall, precision, auc
