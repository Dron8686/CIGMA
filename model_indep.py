import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import random
from torch_geometric.nn import GCNConv
import numpy as np
import math
from parameters import args

# 门控图匹配
class GatedGraphMatcher(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.1):
        super(GatedGraphMatcher, self).__init__()
        self.emb_dim = emb_dim;
        self.hidden_dim = hidden_dim
        self.drug_proj = nn.Linear(emb_dim, emb_dim);
        self.symptom_proj = nn.Linear(emb_dim, emb_dim)
        self.cross_graph_node_embedder = nn.Linear(emb_dim * 2, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.matching_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(hidden_dim, emb_dim))

    def forward(self, drug_neighbors, symptom_neighbors):
        symptom_proj = self.symptom_proj(symptom_neighbors);
        drug_proj = self.drug_proj(drug_neighbors)
        similarity_matrix = torch.bmm(drug_proj, symptom_proj.transpose(1, 2)) / math.sqrt(self.emb_dim)
        attention_ds = F.softmax(similarity_matrix, dim=2);
        drug_context = torch.bmm(attention_ds, symptom_neighbors)
        attention_sd = F.softmax(similarity_matrix.transpose(1, 2), dim=2);
        symptom_context = torch.bmm(attention_sd, drug_neighbors)
        infused_drug = self.cross_graph_node_embedder(torch.cat([drug_neighbors, drug_context], dim=-1))
        infused_symptom = self.cross_graph_node_embedder(torch.cat([symptom_neighbors, symptom_context], dim=-1))
        infused_drug_neighbors = self.norm(infused_drug);
        infused_symptom_neighbors = self.norm(infused_symptom)
        _, drug_gru_hidden = self.gru(infused_drug_neighbors);
        _, symptom_gru_hidden = self.gru(infused_symptom_neighbors)
        drug_vec = torch.cat([drug_gru_hidden[0], drug_gru_hidden[1]], dim=1);
        symptom_vec = torch.cat([symptom_gru_hidden[0], symptom_gru_hidden[1]], dim=1)
        matching_features = torch.cat(
            [drug_vec, symptom_vec, torch.abs(drug_vec - symptom_vec), drug_vec * symptom_vec], dim=1)
        interaction_embedding = self.matching_mlp(matching_features)
        return interaction_embedding


class GraphHSA(BaseModel):
    def __init__(self,
                 protein_num,
                 symptom_num,
                 drug_num,
                 emb_dim,
                 n_hop,
                 l1_decay):
        super(GraphHSA, self).__init__()
        random.seed(args.seed);
        torch.manual_seed(args.seed);
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True;
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        self.protein_num = protein_num
        self.symptom_num = symptom_num
        self.drug_num = drug_num
        self.emb_dim = emb_dim
        self.n_hop = n_hop
        self.l1_decay = l1_decay

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.symptom_embedding = nn.Embedding(self.symptom_num, self.emb_dim)
        self.drug_embedding = nn.Embedding(self.drug_num, self.emb_dim)


        self.batch_drug = nn.BatchNorm1d(self.emb_dim)
        self.batch_sym = nn.BatchNorm1d(self.emb_dim)

        agg_input_dim = self.emb_dim * self.n_hop if self.n_hop > 0 else self.emb_dim
        self.aggregation_function = nn.Linear(agg_input_dim, self.emb_dim)

        self.combine_embedding_pro = nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.batch_pro = nn.BatchNorm1d(self.emb_dim)
        self.combine_embedding_gcn = nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.batch_gcn = nn.BatchNorm1d(self.emb_dim)

        self.gated_matcher = GatedGraphMatcher(emb_dim, hidden_dim=emb_dim)

        self.gcn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for layer in range(args.gcn_layer):
            self.gcn_layers.append(GCNConv(self.emb_dim, self.emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))
            self.dropouts.append(nn.Dropout(0.2))

        self.p_MLP = MLP(self.emb_dim * 3, 16, 1)  # (pro, gcn, interaction)
        self.MLP_pro = MLP_pro(self.emb_dim, 16, 1)
        self.MLP_gcn = MLP_gcn(self.emb_dim, 16, 1)

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def constrative_loss(self, embeds_all, target, temperature=0.1, num_hard_negatives=10):
        target = target.squeeze();
        pos_embeds = embeds_all[target == 1];
        neg_embeds = embeds_all[target == 0]
        if pos_embeds.shape[0] < 2 or neg_embeds.shape[0] < 1: return torch.tensor(0.0, device=embeds_all.device)
        pos_embeds = F.normalize(pos_embeds, p=2, dim=1);
        neg_embeds = F.normalize(neg_embeds, p=2, dim=1);
        anchors = pos_embeds;
        total_loss = 0
        pos_sim_matrix = torch.matmul(anchors, anchors.T) / temperature
        for i in range(anchors.shape[0]):
            pos_mask = torch.ones(anchors.shape[0], dtype=torch.bool, device=anchors.device);
            pos_mask[i] = False;
            if not torch.any(pos_mask): continue
            positive_similarities = pos_sim_matrix[i][pos_mask];
            anchor_neg_similarities = torch.matmul(anchors[i].unsqueeze(0), neg_embeds.T) / temperature
            k = min(num_hard_negatives, neg_embeds.shape[0]);
            hard_negative_similarities, _ = torch.topk(anchor_neg_similarities, k, dim=1)
            all_similarities = torch.cat([positive_similarities.unsqueeze(0), hard_negative_similarities], dim=1);
            log_sum_exp = torch.logsumexp(all_similarities, dim=1)
            loss_per_anchor = -positive_similarities + log_sum_exp;
            total_loss += loss_per_anchor.mean()
        return total_loss / anchors.shape[0] if anchors.shape[0] > 0 else torch.tensor(0.0, device=embeds_all.device)

    def _contrastive_loss_helper(self, embeds1, embeds2):
        logits1 = embeds1 @ embeds2.T / self.temperature;
        logits2 = embeds2 @ embeds1.T / self.temperature
        batch_size = logits1.shape[0];
        labels = torch.arange(batch_size, device=logits1.device)
        loss1 = F.cross_entropy(logits1, labels);
        loss2 = F.cross_entropy(logits2, labels);
        return (loss1 + loss2) / 2.0

    def hybrid_view_loss(self, embeds_pro, embeds_gcn, embeds_interaction, alpha_distill=0.5):
        embeds_pro = F.normalize(embeds_pro, p=2, dim=1);
        embeds_gcn = F.normalize(embeds_gcn, p=2, dim=1);
        embeds_interaction = F.normalize(embeds_interaction, p=2, dim=1)
        loss_pro_gcn = self._contrastive_loss_helper(embeds_pro, embeds_gcn);
        loss_pro_interaction = self._contrastive_loss_helper(embeds_pro, embeds_interaction)
        loss_gcn_interaction = self._contrastive_loss_helper(embeds_gcn, embeds_interaction);
        multi_view_contrastive_loss = (loss_pro_gcn + loss_pro_interaction + loss_gcn_interaction) / 3
        with torch.no_grad(): teacher_embedding = (embeds_pro + embeds_gcn) / 2.0
        student_embedding = embeds_interaction;
        distillation_loss = F.mse_loss(student_embedding, teacher_embedding);
        combined_loss = multi_view_contrastive_loss + alpha_distill * distillation_loss
        return combined_loss

    def forward(self, symptoms: torch.LongTensor, drug: torch.LongTensor,
                sym_feat: torch.LongTensor, dru_feat: torch.LongTensor,
                symptom_neighbors: list, drug_neighbors: list, edge_index: torch.LongTensor):

        device = self.protein_embedding.weight.device

        x_drug = self.drug_embedding(dru_feat.to(device));
        x_drug = self.batch_drug(F.relu(x_drug))
        x_symptom = self.symptom_embedding(sym_feat.to(device));
        symptom_embeddings = x_symptom[symptoms.to(device)];
        drug_embeddings = x_drug[drug.to(device)]
        symptom_neighbors_emb_list = self._get_neighbor_emb(symptom_neighbors, device)
        drug_neighbors_emb_list = self._get_neighbor_emb(drug_neighbors, device)

        symptom_i_list, sym_contributions_raw, sym_contributions_softmax = self._interaction_aggregation(
            symptom_embeddings, symptom_neighbors_emb_list)
        drug_i_list, drug_contributions_raw, drug_contributions_softmax = self._interaction_aggregation(drug_embeddings,
                                                                                                        drug_neighbors_emb_list)

        symptom_agg_embeddings = self._aggregation(symptom_i_list) if self.n_hop > 0 else symptom_embeddings
        drug_agg_embeddings = self._aggregation(drug_i_list) if self.n_hop > 0 else drug_embeddings
        embeds_pro = F.relu(self.combine_embedding_pro(
            torch.cat([drug_agg_embeddings, symptom_agg_embeddings], dim=1)))
        embeds_pro = self.batch_pro(embeds_pro)


        embeds_all_nodes = torch.cat((x_drug, x_symptom))
        for layer in range(args.gcn_layer):
            embeds_all_nodes = self.gcn_layers[layer](embeds_all_nodes.to(device), edge_index.to(device))
            if torch.sum(embeds_all_nodes) != 0: embeds_all_nodes = self.batch_norms[layer](embeds_all_nodes)
            embeds_all_nodes = F.relu(self.dropouts[layer](embeds_all_nodes))
        symptom_embeddings_gcn = embeds_all_nodes[self.drug_num:][symptoms.to(device)]
        drug_embeddings_gcn = embeds_all_nodes[:self.drug_num][drug.to(device)]
        embeds_gcn = F.relu(self.combine_embedding_gcn(
            torch.cat([drug_embeddings_gcn, symptom_embeddings_gcn], dim=1)))
        embeds_gcn = self.batch_gcn(embeds_gcn)

        drug_neighbor_emb_hop1 = drug_neighbors_emb_list[0] if self.n_hop > 0 and drug_neighbors_emb_list else None
        symptom_neighbor_emb_hop1 = symptom_neighbors_emb_list[
            0] if self.n_hop > 0 and symptom_neighbors_emb_list else None
        if drug_neighbor_emb_hop1 is not None and symptom_neighbor_emb_hop1 is not None:
            embeds_interaction = self.gated_matcher(drug_neighbor_emb_hop1, symptom_neighbor_emb_hop1)
        else:
            embeds_interaction = torch.zeros_like(embeds_pro)  # Placeholder

        combined_emb = torch.cat([embeds_pro, embeds_gcn, embeds_interaction], dim=1)

        if args.lambda_sim == 0:
            loss_hybrid = torch.tensor(0.0, device=combined_emb.device)
        else:
            loss_hybrid = self.hybrid_view_loss(embeds_pro, embeds_gcn, embeds_interaction)

        if args.lambda_pred == 0 or args.embed_type != 'pro_gcn':
            score_pro = torch.tensor([], device=combined_emb.device)
            score_gcn = torch.tensor([], device=combined_emb.device)
        else:
            score_pro = self.MLP_pro(embeds_pro);
            score_pro = torch.squeeze(score_pro, 1)
            score_gcn = self.MLP_gcn(embeds_gcn);
            score_gcn = torch.squeeze(score_gcn, 1)

        score = self.p_MLP(combined_emb)
        score = torch.squeeze(score, 1)

        final_att_sym_list = [sym_contributions_softmax[0]] if self.n_hop > 0 and sym_contributions_softmax else []
        final_att_drug_list = [drug_contributions_softmax[0]] if self.n_hop > 0 and drug_contributions_softmax else []

        return score, combined_emb, loss_hybrid, score_pro, score_gcn, final_att_sym_list, final_att_drug_list

    def _get_neighbor_emb(self, neighbors, device):
        neighbors_emb_list = []
        for hop in range(self.n_hop):
            if hop >= len(neighbors): break
            neighbor_indices = neighbors[hop].to(device)
            features = self.protein_embedding(neighbor_indices)
            neighbors_emb_list.append(features)
        return neighbors_emb_list

    def _interaction_aggregation(self, item_embeddings, neighbors_emb_list):
        interact_list = []
        contributions_raw_all_hops = []
        contributions_softmax_all_hops = []
        current_item_embeddings = item_embeddings

        if self.n_hop == 0:
            return interact_list, contributions_raw_all_hops, contributions_softmax_all_hops

        for hop in range(self.n_hop):
            if hop >= len(neighbors_emb_list): break
            neighbor_emb = neighbors_emb_list[hop]
            item_embeddings_expanded = torch.unsqueeze(current_item_embeddings, dim=2)
            contributions = torch.squeeze(torch.matmul(neighbor_emb, item_embeddings_expanded), -1)
            contributions_normalized = F.softmax(contributions, dim=1)
            contributions_raw_all_hops.append(contributions)
            contributions_softmax_all_hops.append(contributions_normalized)
            contributions_expaned = torch.unsqueeze(contributions_normalized, dim=2)
            i = (neighbor_emb * contributions_expaned).sum(dim=1)
            current_item_embeddings = i
            interact_list.append(i)
        return interact_list, contributions_raw_all_hops, contributions_softmax_all_hops

    def _aggregation(self, item_i_list):
        if not item_i_list:
            return None

        item_i_concat = torch.cat(item_i_list, 1)
        expected_dim = self.emb_dim * self.n_hop

        if item_i_concat.shape[1] != expected_dim:
            if item_i_concat.shape[1] > expected_dim:
                item_i_concat = item_i_concat[:, :expected_dim]
            else:
                pad_size = expected_dim - item_i_concat.shape[1]
                padding = torch.zeros(item_i_concat.shape[0], pad_size, device=item_i_concat.device)
                item_i_concat = torch.cat([item_i_concat, padding], dim=1)

        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings


class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output): super(MLP, self).__init__(); self.fc1 = nn.Linear(n_feature,
                                                                                                         n_hidden); self.batch1 = nn.BatchNorm1d(
        n_hidden); self.fc23 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h_1 = torch.tanh(self.fc1(x))
        if x.shape[0] > 1 or not self.training:
            h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x


class MLP_pro(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output): super(MLP_pro, self).__init__(); self.fc1 = nn.Linear(n_feature,
                                                                                                             n_hidden); self.batch1 = nn.BatchNorm1d(
        n_hidden); self.fc23 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h_1 = torch.tanh(self.fc1(x))
        if x.shape[0] > 1 or not self.training:
            h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x


class MLP_gcn(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output): super(MLP_gcn, self).__init__(); self.fc1 = nn.Linear(n_feature,
                                                                                                             n_hidden); self.batch1 = nn.BatchNorm1d(
        n_hidden); self.fc23 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h_1 = torch.tanh(self.fc1(x))
        if x.shape[0] > 1 or not self.training:
            h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x