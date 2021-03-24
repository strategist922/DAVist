from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product, permutations, combinations_with_replacement, chain
import torch.nn.init as init
from torch.autograd import *
import numpy as np
import time
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from models.OrderedAttention import OrderedAttention, FactorGraphAttention

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim_en, hidden_dim_de, projected_size):
        super(AttentionLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim_en, projected_size)
        self.linear2 = nn.Linear(hidden_dim_de, projected_size)
        self.linear3 = nn.Linear(projected_size, 1, False)

    def forward(self, out_e, h):
        '''
        out_e: batch_size * num_frames * en_hidden_dim
        h : batch_size * de_hidden_dim
        '''
        assert out_e.size(0) == h.size(0)
        batch_size, num_frames, _ = out_e.size()
        hidden_dim = h.size(1)

        h_att = h.unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
        x = F.tanh(F.dropout(self.linear1(out_e)) + F.dropout(self.linear2(h_att)))
        x = F.dropout(self.linear3(x))
        a = F.softmax(x.squeeze(2))

        return a


def _smallest(matrix, k, only_first_row=False):
    if only_first_row:
        flatten = matrix[:1, :].flatten()
    else:
        flatten = matrix.flatten()
    args = np.argpartition(flatten, k)[:k]
    args = args[np.argsort(flatten[args])]
    return np.unravel_index(args, matrix.shape), flatten[args]


class VisualEncoder(nn.Module):
    def __init__(self, opt):
        super(VisualEncoder, self).__init__()
        self.opt = opt
        # embedding (input) layer options
        self.feat_size = opt.feat_size
        self.embed_dim = opt.word_embed_dim
        # rnn layer options
        self.rnn_type = opt.visual_rnn_type
        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.visual_dropout
        self.story_size = opt.story_size
        self.with_position = opt.with_position

        # visual feat embedding layer
        self.visual_emb_fc = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                           nn.BatchNorm1d(self.embed_dim),
                                           nn.ReLU(True))
        # visual obj embedding layer
        self.visual_emb_obj = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                            nn.BatchNorm1d(self.embed_dim),
                                            nn.ReLU(True))
        self.semantic_count = 0
        if opt.use_spatial:
            self.spatial_size = opt.spatial_size
            self.visual_emb_obj_spatial = nn.Linear(self.spatial_size, self.embed_dim)
            self.semantic_count += 1
        if opt.use_classes:
            self.num_classes = opt.num_classes
            self.visual_emb_obj_cls = nn.Embedding(self.num_classes, self.embed_dim, padding_idx=0)
            self.semantic_count += 1
        if opt.use_attrs:
            self.num_attrs = opt.num_attrs
            self.visual_emb_obj_attrs = nn.Embedding(self.num_attrs, self.embed_dim, padding_idx=0)
            self.semantic_count += 1
        if self.semantic_count > 1:
            self.semantic_fusion = nn.Sequential(nn.Linear(self.embed_dim * self.semantic_count, self.embed_dim),
                                                 nn.BatchNorm1d(self.embed_dim),
                                                 nn.ReLU(True))
        self.dropout1 = nn.Dropout(self.dropout)

        # # factor graph attention
        # self.util_e = [self.embed_dim for _ in range(self.story_size)]
        # self.fga = OrderedAttention(self.util_e, shared_weights=[0, 1, 2, 3, 4], share_self=False, share_bias=False, sizes=[None], pairwise_flag=False)
        # # self.fga_layer_1 = OrderedAttention(self.util_e, shared_weights=[0, 1, 2, 3, 4], share_self=False, share_bias=False, sizes=[5])
        # self.fga2 = FactorGraphAttention(self.util_e, self_flag=True, unary_flag=True,\
        #                                             pairwise_flag=False, share_bias=False, share_self=False, sizes=[5])

        # embedding fusion layer
        self.fuse_emb = nn.Sequential(nn.Linear(self.embed_dim * 2, self.embed_dim),
                                            nn.BatchNorm1d(self.embed_dim),
                                            nn.ReLU(True))

        # visual rnn layer
        # self.hin_dropout_layer = nn.Dropout(self.dropout)
        # transformer_layer = TransformerEncoderLayer(self.embed_dim, opt.num_heads)
        # self.rnn = TransformerEncoder(transformer_layer, opt.num_layers)

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU()
        self.position_vec = torch.arange(0,5, requires_grad=False).cuda()
        self.position_embed_fga = nn.Embedding(self.story_size, self.embed_dim)
        self.position_embed_context = nn.Embedding(self.story_size, self.embed_dim)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return weight.new(self.num_layers * times, batch_size, dim).zero_()
        else:
            return (weight.new(self.num_layers * times, batch_size, dim).zero_(),
                    weight.new(self.num_layers * times, batch_size, dim).zero_())

    def forward(self, input_feat, input_obj, hidden=None, spatial=None, clss=None, attrs=None):
        """
        inputs:
        - input_feat  	(batch_size, 5, feat_size)
        - input_feat  	(batch_size, 5, num_obj, feat_size)
        - hidden 	(num_layers * num_dirs, batch_size, hidden_dim // 2)
        return:
        - out 		(batch_size, 5, rnn_size), serve as context
        """
        batch_size, seq_length, num_obj = input_obj.size(0), input_obj.size(1), input_obj.size(2)

        # visual embeded feat
        emb_fc = self.visual_emb_fc(input_feat.view(-1, self.feat_size))
        emb_fc = emb_fc.view(batch_size, seq_length, -1)  # (Na, album_size, embedding_size)

        # visual embeded obj
        emb_obj = self.visual_emb_obj(input_obj.view(-1, self.feat_size))
        if self.semantic_count > 0:
            emb_sem = []
            if self.opt.use_spatial:
                emb_sem.append(self.visual_emb_obj_spatial(spatial.view(-1, self.spatial_size)))
            if self.opt.use_classes:
                emb_sem.append(self.visual_emb_obj_cls(clss.view(-1)))
            if self.opt.use_attrs:
                emb_sem.append(self.visual_emb_obj_attrs(attrs.view(-1)))
            emb_sem = torch.cat(emb_sem, dim=1)
            if self.semantic_count > 1:
                emb_sem = self.semantic_fusion(emb_sem)
            emb_obj = emb_obj + emb_sem
            emb_obj = self.dropout1(emb_obj)
        emb_obj = emb_obj.view(batch_size, seq_length, num_obj, -1)  # (Na, album_size, embedding_size)
        # emb_obj, _ = self.fga([img.squeeze(1) for img in torch.split(emb_obj, 1, dim=1)])
        # emb_obj, _ = self.fga2([torch.stack(x,1) for x in emb_obj])
        # emb_obj = torch.stack(emb_obj, dim=1)
        # emb_obj = torch.mean(emb_obj, dim=2)
        emb_obj = torch.mean(emb_obj, dim=2)
        # fuse features
        emb = self.fuse_emb(torch.cat((emb_fc, emb_obj), dim=2).view(-1, self.embed_dim*2))
        emb = self.relu(emb)
        emb = emb.view(batch_size, seq_length, -1)
        emb = emb + self.position_embed_context(self.position_vec)

        # # visual rnn layer
        # rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        # rnn_input = rnn_input.transpose(1, 0) # (5, batch_size, embed_dim)
        # emb_trans = self.rnn(rnn_input).transpose(1, 0) # (batch_size, 5, embed_dim)
        # emb = emb + self.relu(emb_trans).contiguous()  # (batch_size, 5, embed_dim)
        return emb, hidden


def weight_init(m):
    '''
    adapted from https://gist.github.com/jeasinema/"
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def create_occurance_matrix(m, vocab_size):
    counter = np.zeros((m.shape[0],vocab_size))
    for i in range(m.shape[0]):
        ind, count = np.unique(m[i], return_counts=True)
        counter[i][ind] = count
    return counter
