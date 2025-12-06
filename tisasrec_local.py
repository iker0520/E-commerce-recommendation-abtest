# -*- coding: utf-8 -*-
# tisasrec_local.py (최종 수정본)

import torch
import torch.nn as nn
import numpy as np

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

class TiSASRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(TiSASRec, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.time_span = config["time_span"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.time_matrix_K_emb = torch.nn.Embedding(self.time_span + 1, self.hidden_size)
        self.time_matrix_V_emb = torch.nn.Embedding(self.time_span + 1, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def compute_time_matrix(self, time_stamp):
        seq_len = time_stamp.shape[1]
        time_stamp = time_stamp.float()
        timestamp_matrix = time_stamp.unsqueeze(1)
        timestamp_matrix_repeat = timestamp_matrix.repeat(1, seq_len, 1)
        diff_matrix = timestamp_matrix_repeat.transpose(-1, -2) - timestamp_matrix_repeat
        return torch.clamp(torch.abs(diff_matrix).int(), max=self.time_span)

    # [tisasrec_local.py 내부의 TiSASRec 클래스]

    def forward(self, item_seq, item_seq_len, time_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        time_matrix = self.compute_time_matrix(time_seq)
        time_key = self.dropout(self.time_matrix_K_emb(time_matrix))
        time_value = self.dropout(self.time_matrix_V_emb(time_matrix))

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # ------------------------------------------------------------------
        # [수정된 부분] 기존 self.get_attention_mask(item_seq) 삭제하고 아래로 교체
        # ------------------------------------------------------------------
        # 1. 패딩(0)인 부분은 True, 아니면 False인 마스크 생성
        mask = (item_seq == 0) 
        
        # 2. 차원 확장 [Batch, 1, 1, SeqLen] (헤드와 시퀀스 차원에 맞춤)
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        
        # 3. 확실하게 Boolean 타입으로 지정 (에러 해결 핵심)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.bool)
        # ------------------------------------------------------------------

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, time_key, time_value, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction['timestamp_list']
        
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


# --- Helper Classes ---

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList([
            TiTransformerLayer(hidden_size, n_heads, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states, attention_mask, time_key, time_value, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, time_key, time_value)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class TiTransformerLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(TiTransformerLayer, self).__init__()
        self.multi_head_attention = TiMultiHeadAttention(hidden_size, n_heads, attn_dropout_prob)
        self.feed_forward = FeedForward(hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        # [수정됨] 여기에 있던 self.LayerNorm 삭제함 (체크포인트에 없음)

    def forward(self, hidden_states, attention_mask, time_key, time_value):
        attention_output = self.multi_head_attention(hidden_states, attention_mask, time_key, time_value)
        feed_forward_output = self.feed_forward(attention_output)
        return feed_forward_output

class TiMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, attn_dropout_prob):
        super(TiMultiHeadAttention, self).__init__()
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.out_dropout = nn.Dropout(attn_dropout_prob)

    def forward(self, hidden_states, attention_mask, time_key, time_value):
        B, L, _ = hidden_states.size()
        Q = self.query(hidden_states).view(B, L, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(B, L, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(B, L, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

        time_key = time_key.view(B, L, L, self.num_attention_heads, self.attention_head_size).permute(0, 3, 1, 2, 4)
        time_value = time_value.view(B, L, L, self.num_attention_heads, self.attention_head_size).permute(0, 3, 1, 2, 4)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores += torch.sum(Q.unsqueeze(3) * time_key, dim=-1)
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)

        if attention_mask is not None:
             attention_scores = attention_scores.masked_fill(attention_mask, -1e9)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, V)
        context_layer += torch.sum(attention_probs.unsqueeze(-1) * time_value, dim=-2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        output = self.dense(context_layer)
        output = self.out_dropout(output)
        output = self.LayerNorm(output + hidden_states)
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self._get_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def _get_act(self, act):
        if act == 'relu': return nn.ReLU()
        if act == 'gelu': return nn.GELU()
        return nn.GELU()

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states