# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/15/2020
#

"""
  Disentangled SelfAttention module
"""

import numpy as np
import torch
import math
from .ops import *

__all__=['build_relative_position', 'DisentangledSelfAttention']

def build_relative_position(query_size, key_size):
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
    rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


class DisentangledSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj = torch.nn.Linear(config.hidden_size, self.all_head_size*3, bias=False)
        self.q_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'none').lower().split('|')] # c2p|p2c
        
        self.relative_attention = getattr(config, 'relative_attention', False)
        self.talking_head = getattr(config, 'talking_head', False)

        if self.talking_head:
            self.head_logits_proj = torch.nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = torch.nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions <1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_proj = torch.nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_q_proj = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        if query_states is None:
            qp = self.in_proj(hidden_states) #.split(self.all_head_size, dim=-1)
            query_layer,key_layer,value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:
            def linear(w,b,x):
                if b is not None:
                    return torch.matmul(x, w.t()) + b.t()
                else:
                    return torch.matmul(x, w.t()) # + b.t()

            ws = self.in_proj.weight.chunk(self.num_attention_heads*3, dim=0)
            qkvw = [torch.cat([ws[i*3+k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            qkvb = [None]*3

            q = linear(qkvw[0], qkvb[0], query_states)
            k,v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1,3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q,k,v]]

        query_layer += self.transpose_for_scores(self.q_bias.unsqueeze(0).unsqueeze(0))
        value_layer += self.transpose_for_scores(self.v_bias.unsqueeze(0).unsqueeze(0))

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.size(-1)*scale_factor)
        query_layer = query_layer/scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)

        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.permute(0,2,3,1)).permute(0,3,1,2)

        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.permute(0,2,3,1)).permute(0,3,1,2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_att:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2))
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[self.max_relative_positions - att_span:self.max_relative_positions + att_span, :].unsqueeze(0)
        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2)).to(query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
            score += p2c_att

        return score
