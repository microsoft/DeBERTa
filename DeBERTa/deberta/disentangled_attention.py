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
import math
import torch
from torch import nn
import functools
import pdb

from .ops import *
from .da_utils import build_relative_position

from ..utils import get_logger
logger=get_logger()

__all__=['DisentangledSelfAttention']
class DisentangledSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.attention_head_size = getattr(config, 'attention_head_size', _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, 'share_att_key', False)
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'c2p').lower().split('|')] # c2p|p2c
        self.relative_attention = getattr(config, 'relative_attention', False)

        if self.relative_attention:
            self.position_buckets = getattr(config, 'position_buckets', -1)
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions <1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets>0:
                self.pos_ebd_size = self.position_buckets
                # For backward compitable

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if (not self.share_att_key):
                if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)
        self._register_load_state_dict_pre_hook(self._pre_load_hook)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads).float()
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads).float()
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        
        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = 1/math.sqrt(query_layer.size(-1)*scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)*scale)
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)
        attention_scores = (attention_scores - attention_scores.max(dim=-1, keepdim=True).values.detach()).to(hidden_states)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))

        # bxhxlxd
        _attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(_attention_probs)
        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return {
            'hidden_states': context_layer,
            'attention_probs': _attention_probs,
            'attention_logits': attention_scores
            }

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = 1/math.sqrt(pos_key_layer.size(-1)*scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2).to(query_layer)*scale)
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = 1/math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2).to(key_layer)*scale)
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
            score += p2c_att

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:,:,att_span:,:]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
            p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += p2p_att

        return score

    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
        self_state = self.state_dict()
        if ((prefix + 'query_proj.weight') not in state_dict) and ((prefix + 'in_proj.weight') in state_dict):
          v1_proj = state_dict[prefix+'in_proj.weight']
          v1_proj = v1_proj.unsqueeze(0).reshape(self.num_attention_heads, -1, v1_proj.size(-1))
          q,k,v=v1_proj.chunk(3, dim=1)
          state_dict[prefix + 'query_proj.weight'] = q.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.weight'] = k.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.bias'] = self_state['key_proj.bias']
          state_dict[prefix + 'value_proj.weight'] = v.reshape(-1, v1_proj.size(-1))
          v1_query_bias = state_dict[prefix + 'q_bias']
          state_dict[prefix + 'query_proj.bias'] = v1_query_bias
          v1_value_bias = state_dict[prefix +'v_bias']
          state_dict[prefix + 'value_proj.bias'] = v1_value_bias

          v1_pos_key_proj = state_dict[prefix + 'pos_proj.weight']
          state_dict[prefix + 'pos_key_proj.weight'] = v1_pos_key_proj
          v1_pos_query_proj = state_dict[prefix + 'pos_q_proj.weight']
          state_dict[prefix + 'pos_query_proj.weight'] = v1_pos_query_proj
          v1_pos_query_proj_bias = state_dict[prefix + 'pos_q_proj.bias']
          state_dict[prefix + 'pos_query_proj.bias'] = v1_pos_query_proj_bias
          state_dict[prefix + 'pos_key_proj.bias'] = self_state['pos_key_proj.bias']

          del state_dict[prefix + 'in_proj.weight']
          del state_dict[prefix + 'q_bias']
          del state_dict[prefix + 'v_bias']
          del state_dict[prefix + 'pos_proj.weight']
          del state_dict[prefix + 'pos_q_proj.weight']
          del state_dict[prefix + 'pos_q_proj.bias']
