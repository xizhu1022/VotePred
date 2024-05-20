import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from torch_geometric.nn import RGCNConv


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, dim1 // 2)
        self.fc2 = torch.nn.Linear(dim1 // 2, 1)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = x.float()
        h1 = self.act(self.fc1(x))
        return self.fc2(h1)


class RGCNModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCNModel, self).__init__()
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations)
        self.fc = nn.Linear(out_channels, 2)  # Assuming binary classification task

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x

#
# class DualAttention(nn.Module):
#     def __init__(self, d_model, d_k, d_v, n_head, dropout):
#         super(DualAttention, self).__init__()
#         self.dropout = dropout
#         self.n_head = n_head
#         self.d_model = d_model
#         self.left_attention = nn.MultiheadAttention(embed_dim= d_model, num_heads=n_head, dropout=dropout)
#         self.right_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
#
#     def forward(self, bill_emb, sponsor_emb, topic_emb):
#         sponsor_emb, sponsor_dist = self.left_attention(query=bill_emb, key=sponsor_emb, value=sponsor_emb,
#                                           key_padding_mask=False, need_weights=True, attn_mask=None)
#
#         topic_emb, topic_dist = self.right_attention(query=bill_emb, key=topic_emb, value=topic_emb,
#                                                         key_padding_mask=False, need_weights=True, attn_mask=None)
#         # fusion


class rgcn_bert(nn.Module):
    def __init__(self, num_users, embedding_size, hidden_size, output_size):
        super(rgcn_bert, self).__init__()
        self.num_users = num_users
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        num_relations = 5

        self.init_emb()
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.RGCNModel = RGCNModel(self.embedding_size, 64, num_relations)
        self.affinity_score = MergeLayer(128)

        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)

    def cal_loss(self, pos_pre, neg_pre):
        # pred: [bs, 1+neg_num]
        targets = torch.cat([torch.ones_like(pos_pre), torch.zeros_like(pos_pre)], dim=0)
        predictions = torch.cat([pos_pre, neg_pre], dim=0)
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets)
        predicted_labels = torch.round(torch.sigmoid(predictions))

        correct = (predicted_labels == targets).sum().item()
        accuracy = correct / targets.size(0)
        f1 = f1_score(targets.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        recall = recall_score(targets.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        precision = precision_score(targets.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        auc = roc_auc_score(targets.detach().cpu().numpy(), torch.sigmoid(predictions).detach().cpu().numpy())

        return bce_loss, accuracy, f1, recall, precision, auc

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

        bce_loss, accuracy, f1, recall, precision, auc = self.cal_loss(pos_pre, neg_pre)

        return bce_loss, accuracy, f1, recall, precision, auc
