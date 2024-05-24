import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

from utils import cal_loss


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

    def forward(self, left_embeddings, right_embeddings, legistor_embeddings):
        x = torch.cat([left_embeddings, right_embeddings, legistor_embeddings], dim=1)
        x = x.float()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


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
    def __init__(self, d_model, num_heads, dropout):
        super(DualAttention, self).__init__()
        self.dropout = dropout
        self.n_head = num_heads
        self.d_model = d_model

        self.left_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head, dropout=self.dropout)
        self.right_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head, dropout=self.dropout)

        # self.bill2cosponsers = data.bill2cosponsers
        # self.bill2subjects = data.bill2subjects

    def forward(self, node_embeddings, bill_idx, sponser_idx, subject_idx, sponser_masks, subject_masks):
        bill_embeddings = node_embeddings[bill_idx].unsqueeze(0)  # (1, bsz, dim)
        # sponsers = self.bill2cosponsers[bill_index]
        # subjects = self.bill2subjects[bill_index]
        sponser_embeddings = node_embeddings[sponser_idx].transpose(0, 1)  # (num_sponsers, bsz, dim)
        subject_embeddings = node_embeddings[subject_idx].transpose(0, 1)  # (num_subjects, bsz, dim)

        left_embeddings, left_weights = self.left_attn(query=bill_embeddings,
                                                       key=sponser_embeddings,
                                                       value=sponser_embeddings,
                                                       key_padding_mask=False,
                                                       need_weights=True,
                                                       attn_mask=sponser_masks)

        right_embeddings, right_weights = self.right_attn(query=bill_embeddings,
                                                          key=subject_embeddings,
                                                          value=subject_embeddings,
                                                          key_padding_mask=False,
                                                          need_weights=True,
                                                          attn_mask=subject_masks)
        left_embeddings = left_embeddings.squeeze(0)  # (bsz, dim)
        right_embeddings = right_embeddings.squeeze(0)  # (bsz, dim)

        return left_embeddings, right_embeddings


class RGCN_DualAttn_FFNN(nn.Module):
    def __init__(self,
                 dim,
                 num_nodes,
                 num_relations,
                 edge_index,
                 edge_type,
                 num_heads,
                 dropout):
        super(RGCN_DualAttn_FFNN, self).__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.n_head = num_heads
        self.dropout = dropout

        self.node_embeddings = nn.Parameter(torch.FloatTensor(self.num_nodes, self.dim))

        self.RGCNModel = RGCNModel(in_channels=self.dim,
                                   out_channels=self.dim,
                                   num_relations=self.num_relations)

        self.DualAttn = DualAttention(d_model=self.dim,
                                      num_heads=self.n_head,
                                      dropout=self.dropout)

        self.FFNN = FFNN(input_dim=self.dim * 3)

    def forward(self, legis_idx, bill_idx, sponser_idx, subject_idx, sponser_masks, subject_masks):
        # TODO: Check inputs
        node_embeddings = self.RGCNModel(x=self.node_embeddings)
        legis_embeddings = node_embeddings[legis_idx]
        left_embeddings, right_embeddings = self.DualAttn(node_embeddings=node_embeddings,
                                                          bill_idx=bill_idx,
                                                          sponser_idx=sponser_idx,
                                                          subject_idx=subject_idx,
                                                          sponser_masks=sponser_masks,
                                                          subject_masks=subject_masks)
        scores = self.FFNN(left_embeddings=left_embeddings,
                           right_embeddings=right_embeddings,
                           legistor_embeddings=legis_embeddings)

        scores = scores.squeeze(-1).cpu().tolist()
        pos_scores = scores[: len(scores)//2]
        neg_scores = scores[len(scores)//2:]

        return cal_loss(pos_scores, neg_scores)


class RGCN_Merge(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, output_size, num_relations):
        super(RGCN_Merge, self).__init__()
        self.num_nodes = num_nodes  # all nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_relations = num_relations

        self.node_embeddings = nn.Parameter(torch.FloatTensor(self.num_nodes, self.input_size))

        self.RGCNModel = RGCNModel(self.input_size, 64, self.num_relations)
        self.affinity_score = MergeLayer(128)

        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.node_embeddings)

    def forward(self, batch, device, edge_index_combined, edge_type_combined):
        _, info = batch
        bill_id, user1_id, user2_id = info
        bill_id = bill_id.to(device)
        user1_id = user1_id.to(device)
        user2_id = user2_id.to(device)

        edge_index_combined = edge_index_combined.to(device)
        edge_type_combined = edge_type_combined.to(device)
        nodes_embeddings = self.RGCNModel(self.users_feature, edge_index_combined, edge_type_combined).to(device)

        user1_embedding = nodes_embeddings[user1_id]  # pos
        user2_embedding = nodes_embeddings[user2_id]  # neg
        bill_embedding = nodes_embeddings[bill_id]

        pos_pre = self.affinity_score(bill_embedding, user1_embedding)
        neg_pre = self.affinity_score(bill_embedding, user2_embedding)

        bce_loss, accuracy, f1, recall, precision, auc = cal_loss(pos_pre, neg_pre)

        return bce_loss, accuracy, f1, recall, precision, auc

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
