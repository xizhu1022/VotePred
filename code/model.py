import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

import random

from hgb import myGAT


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
                 if_pre_train,
                 fusion_type,
                 encoder_type,
                 data):
        super(RGCN_DualAttn_FFNN, self).__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.negative_slope = negative_slope
        self.data = data
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha

        self.if_pre_train = if_pre_train
        self.fusion_type = fusion_type
        self.encoder_type = encoder_type

        if self.if_pre_train:
            self.node_embeddings = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=768)
            self.pretrained_embeddings = self.data.load_pretrained_embeddings()
            self.EmbeddingMLP = nn.Linear(768, self.dim)

        else:
            self.node_embeddings = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.dim)

        if self.encoder_type == 'hgb':
            self.Encoder = myGAT(
                edge_dim=self.dim,
                num_etypes=self.num_relations,
                in_dims=[self.dim],
                num_hidden=self.dim,
                num_layers=self.num_layers,
                heads=[self.num_heads] * self.num_layers,
                activation=F.relu,
                feat_drop=self.dropout_1,
                attn_drop=self.dropout_1,
                negative_slope=self.negative_slope,
                residual=False,
                alpha=self.alpha)

        elif self.encoder_type == 'rgcn':
            self.Encoder = RGCN(
                in_channels=self.dim,
                out_channels=self.dim,
                num_relations=self.num_relations,
                num_layers=self.num_layers)

        elif self.encoder_type == 'none':
            pass

        else:
            raise NotImplementedError

        self.DualAttn = DualAttention(d_model=self.dim,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout_2)

        self.FusionLayer = Fusion(dim=self.dim,
                                  num_heads=self.num_heads,
                                  fusion_type=self.fusion_type)

        self.PredictorLayer = Predictor()

        self.initialize()

    def initialize(self):
        if self.if_pre_train:
            self.node_embeddings.weight.data.copy_(self.pretrained_embeddings)
            nn.init.normal_(self.EmbeddingMLP.weight, mean=0, std=0.01)
            nn.init.constant_(self.EmbeddingMLP.bias, 0)
        else:
            nn.init.xavier_normal_(self.node_embeddings.weight.data)

    def forward(self, batch, graph):
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

        if self.if_pre_train:
            if isinstance(self.node_embeddings, nn.Embedding):
                self.node_embeddings = self.node_embeddings.weight
            x = self.EmbeddingMLP(self.node_embeddings)
        else:
            x = self.node_embeddings

        if self.encoder_type == 'hgb':
            node_embeddings, self.encoder_weights = self.Encoder(features_list=[x],
                                                                 e_feat=graph.edata['etype'],
                                                                 g=graph)

        elif self.encoder_type == 'rgcn':
            node_embeddings = self.Encoder(x=x,
                                           edge_indexes=graph.edges(),
                                           edge_types=graph.edata['etype'])

        elif self.encoder_type == 'none':
            node_embeddings = x
        else:
            raise NotImplementedError

        legis_embeddings = node_embeddings[mid_batch]  # (bsz, dim)
        pos_bill_embeddings = node_embeddings[pos_bill_index_batch]
        neg_bill_embeddings = node_embeddings[neg_bill_index_batch]

        pos_left_embeddings, pos_right_embeddings = self.DualAttn(node_embeddings=node_embeddings,
                                                                  query_idx=mid_batch,
                                                                  sponser_idx=pos_cosponser_batch,
                                                                  subject_idx=pos_subject_batch,
                                                                  sponser_masks=pos_cosponser_masks,
                                                                  subject_masks=pos_subject_masks)  # (bsz, dim)

        pos_embeddings = self.FusionLayer(bill_embeddings=pos_bill_embeddings,
                                          left_embeddings=pos_left_embeddings,
                                          right_embeddings=pos_right_embeddings)

        neg_left_embeddings, neg_right_embeddings = self.DualAttn(node_embeddings=node_embeddings,
                                                                  query_idx=mid_batch,
                                                                  sponser_idx=neg_cosponser_batch,
                                                                  subject_idx=neg_subject_batch,
                                                                  sponser_masks=neg_cosponser_masks,
                                                                  subject_masks=neg_subject_masks)

        neg_embeddings = self.FusionLayer(bill_embeddings=neg_bill_embeddings,
                                          left_embeddings=neg_left_embeddings,
                                          right_embeddings=neg_right_embeddings)

        pos_scores = self.PredictorLayer(bill_embeddings=pos_embeddings,
                                         legislator_embeddings=legis_embeddings)  # (bsz)

        neg_scores = self.PredictorLayer(bill_embeddings=neg_embeddings,
                                         legislator_embeddings=legis_embeddings)  # (bsz)

        return self.cal_loss(pos_scores, neg_scores, node_embeddings)

    def cal_sim_loss(self, pairs, batch_size, node_embeddings):
        # valid_indices = list(set(list(pairs[0]) + list(pairs[1])))

        valid_indices = [self.data.node2index[x] for x in self.data.mid_list]
        sample_indices = random.sample(list(np.arange(len(pairs[0]))), k=batch_size)

        fro = list(pairs[0][sample_indices])
        to = list(pairs[1][sample_indices])
        another = random.sample(valid_indices, k=batch_size)

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
            legislator = random.sample(self.data.train_mids, k=1)[0]  # np.random.choice(self.data.train_mids)
            candidate_pos_bills = self.data.train_mid2results[legislator]['proposals']
            candidate_neg_bills = self.data.train_mid2results[legislator]['yeas']

            if len(candidate_pos_bills) == 0:
                continue
            else:
                pos_bill = random.sample(candidate_pos_bills, k=1)[0]

            if len(candidate_neg_bills) == 0:
                continue
            else:
                neg_bill = random.sample(candidate_neg_bills, k=1)[0]  # np.random.choice(candidate_neg_bills)

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

        # main_loss: roll call prediction
        targets = torch.cat([torch.ones_like(pos_pre), torch.zeros_like(pos_pre)], dim=0)
        predictions = torch.cat([pos_pre, neg_pre], dim=0)
        main_loss = F.binary_cross_entropy_with_logits(predictions, targets)

        # loss_1 group similarity committee / twitter / state / party
        loss_1 = 0

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

        predictions = torch.sigmoid(predictions)
        predicted_labels = torch.round(predictions)
        targets = targets.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        predicted_labels = predicted_labels.detach().cpu().numpy()

        return loss, targets, predictions, predicted_labels


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


class Fusion(nn.Module):
    def __init__(self, dim, num_heads, fusion_type):
        super(Fusion, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.fusion_type = fusion_type

        if self.fusion_type == 'mean':
            pass

        elif self.fusion_type == 'concat_mlp':
            self.fc = nn.Linear(in_features=self.dim * 3, out_features=self.dim)

        elif self.fusion_type == 'concat2_self_attn_mlp':
            self.fusion_attn = nn.MultiheadAttention(embed_dim=self.dim * 2, num_heads=self.num_heads)
            self.fc = nn.Linear(in_features=self.dim * 2, out_features=self.dim)

        elif self.fusion_type == 'concat3_self_attn_mlp':
            self.fusion_attn = nn.MultiheadAttention(embed_dim=self.dim * 3, num_heads=self.num_heads)
            self.fc = nn.Linear(in_features=self.dim * 3, out_features=self.dim)

        elif self.fusion_type == 'self_attn_mean_mlp':
            self.left_attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads)
            self.right_attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads)
            self.bill_attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads)
            self.fc = nn.Linear(in_features=self.dim, out_features=self.dim)

        elif self.fusion_type == 'ablation_wo_both':
            self.fusion_attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads)
            self.fc = nn.Linear(in_features=self.dim, out_features=self.dim)

        elif self.fusion_type in ['ablation_only_sponsers', 'ablation_only_subjects']:
            self.fusion_attn = nn.MultiheadAttention(embed_dim=self.dim * 2, num_heads=self.num_heads)
            self.fc = nn.Linear(in_features=self.dim * 2, out_features=self.dim)

    def initialize(self):
        if self.fusion_type in ['concat_mlp', 'self_attn_mean_mlp', 'concat2_self_attn_mlp', 'concat3_self_attn_mlp']:
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, bill_embeddings, left_embeddings, right_embeddings):
        if self.fusion_type == 'mean':
            x = torch.stack([left_embeddings, bill_embeddings, right_embeddings], dim=-1)
            x = torch.mean(x, dim=-1, keepdim=False)
            return x

        elif self.fusion_type == 'concat_mlp':
            x = torch.cat([left_embeddings, bill_embeddings, right_embeddings], dim=1)
            x = self.fc(x)
            return x

        elif self.fusion_type == 'concat3_self_attn_mlp':
            x = torch.cat([left_embeddings, bill_embeddings, right_embeddings], dim=1)
            x = torch.unsqueeze(x, dim=0)
            x, _ = self.fusion_attn(query=x, key=x, value=x)
            x = torch.squeeze(x, dim=0)
            x = self.fc(x)
            return x

        elif self.fusion_type == 'concat2_self_attn_mlp':
            x = torch.cat([left_embeddings, right_embeddings], dim=1)
            x = torch.unsqueeze(x, dim=0)
            x, _ = self.fusion_attn(query=x, key=x, value=x)
            x = torch.squeeze(x, dim=0)
            x = self.fc(x)
            return x

        elif self.fusion_type == 'self_attn_mean_mlp':
            left_embeddings = torch.unsqueeze(left_embeddings, dim=0)
            bill_embeddings = torch.unsqueeze(bill_embeddings, dim=0)
            right_embeddings = torch.unsqueeze(right_embeddings, dim=0)

            left_embeddings, _ = self.left_attn(query=left_embeddings, key=left_embeddings, value=left_embeddings)
            bill_embeddings, _ = self.bill_attn(query=bill_embeddings, key=bill_embeddings, value=bill_embeddings)
            right_embeddings, _ = self.right_attn(query=right_embeddings, key=right_embeddings, value=right_embeddings)

            left_embeddings = torch.squeeze(left_embeddings, dim=0)
            bill_embeddings = torch.squeeze(bill_embeddings, dim=0)
            right_embeddings = torch.squeeze(right_embeddings, dim=0)

            x = torch.stack([left_embeddings, bill_embeddings, right_embeddings], dim=-1)
            x = torch.mean(x, dim=-1, keepdim=False)
            x = self.fc(x)
            return x

        elif self.fusion_type == 'ablation_wo_both':
            x = bill_embeddings
            x = torch.unsqueeze(x, dim=0)
            x, _ = self.fusion_attn(query=x, key=x, value=x)
            x = torch.squeeze(x, dim=0)
            x = self.fc(x)
            return x

        elif self.fusion_type in ['ablation_only_sponsers', 'ablation_only_subjects']:
            if self.fusion_type == 'ablation_only_sponsers':
                x = torch.cat([left_embeddings, bill_embeddings], dim=1)
            elif self.fusion_type == 'ablation_only_subjects':
                x = torch.cat([bill_embeddings, right_embeddings], dim=1)
            else:
                raise NotImplementedError
            x = torch.unsqueeze(x, dim=0)
            x, _ = self.fusion_attn(query=x, key=x, value=x)
            x = torch.squeeze(x, dim=0)
            x = self.fc(x)
            return x


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

    def forward(self, bill_embeddings, legislator_embeddings):
        scores = (bill_embeddings * legislator_embeddings).sum(axis=1)
        return scores


class RGCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_layers):
        super(RGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_layers = num_layers

        self.rgcn_layers = nn.ModuleList()
        self.rgcn_layers.append(RGCNConv(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         num_relations=self.num_relations))

        for l in range(1, self.num_layers):
            self.rgcn_layers.append(RGCNConv(in_channels=self.out_channels,
                                             out_channels=self.out_channels,
                                             num_relations=self.num_relations))

        self.activation = nn.ReLU()

    def forward(self, x, edge_indexes, edge_types):
        if isinstance(x, nn.Embedding):
            x = x.weight
        edge_indexes = torch.stack(edge_indexes, dim=0)
        for l in range(self.num_layers - 1):
            x = self.rgcn_layers[l](x=x, edge_index=edge_indexes, edge_type=edge_types)
            x = self.activation(x)
        x = self.rgcn_layers[-1](x=x, edge_index=edge_indexes, edge_type=edge_types)
        return x


class DualAttention(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 dropout):
        super(DualAttention, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.d_model = d_model

        self.left_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads, dropout=self.dropout)
        self.right_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads, dropout=self.dropout)

    def forward(self, node_embeddings, query_idx, sponser_idx, subject_idx, sponser_masks, subject_masks):
        query_embeddings = node_embeddings[query_idx].unsqueeze(0)  # (1, bsz, dim)
        sponser_embeddings = node_embeddings[sponser_idx].transpose(0, 1)  # (num_sponsers, bsz, dim)
        subject_embeddings = node_embeddings[subject_idx].transpose(0, 1)  # (num_subjects, bsz, dim)

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
