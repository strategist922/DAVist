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

        # factor graph attention
        self.util_e = [self.embed_dim for _ in range(self.story_size)]
        self.fga = FactorGraphAttention(self.util_e, shared_weights=[0, 1, 2, 3, 4], share_self=True)

        # embedding fusion layer
        self.fuse_emb = nn.Sequential(nn.Linear(self.embed_dim * 2, self.embed_dim),
                                            nn.BatchNorm1d(self.embed_dim),
                                            nn.ReLU(True))

        # visual rnn layer
        self.hin_dropout_layer = nn.Dropout(self.dropout)
        transformer_layer = TransformerEncoderLayer(self.embed_dim, opt.num_heads)
        self.rnn = TransformerEncoder(transformer_layer, opt.num_layers)

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.relu = nn.ReLU()

        if self.with_position:
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

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
        emb_obj, attn_weights = self.fga([img.squeeze(1) for img in torch.split(emb_obj, 1, dim=1)])
        emb_obj = torch.stack(emb_obj, dim=1)
        # emb_obj = torch.mean(emb_obj, dim=2)

        # fuse features
        emb = self.fuse_emb(torch.cat((emb_fc, emb_obj), dim=2).view(-1, self.embed_dim*2))
        emb = emb.view(batch_size, seq_length, -1)

        # # visual rnn layer
        rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        rnn_input = rnn_input.transpose(1, 0) # (5, batch_size, embed_dim)
        emb_trans = self.rnn(rnn_input).transpose(1, 0) # (batch_size, 5, embed_dim)
        emb = emb + self.relu(emb_trans).contiguous()  # (batch_size, 5, embed_dim)
        return emb, hidden


class Unary(nn.Module):
    def __init__(self, embed_size):
        super(Unary, self).__init__()
        self.embed = nn.Conv1d(embed_size, embed_size, 1)
        self.feature_reduce = nn.Conv1d(embed_size, 1, 1)
        self.dropout = nn.Dropout()

    def forward(self, X):
        X = X.transpose(1, 2)
        X_embed = self.embed(X)
        X_nl_embed = F.relu(X_embed)
        X_nl_embed = self.dropout(X_nl_embed)
        X_poten = self.feature_reduce(X_nl_embed)
        return X_poten.squeeze(1)


class Pairwise(nn.Module):
    def __init__(self, embed_x_size, x_spatial_dim=None, embed_y_size=None, y_spatial_dim=None):
        super(Pairwise, self).__init__()
        # print(x_spatial_dim, y_spatial_dim)
        embed_y_size = embed_y_size if embed_y_size is not None else embed_x_size
        self.y_spatial_dim = y_spatial_dim if y_spatial_dim is not None else x_spatial_dim

        self.embed_size = max(embed_x_size, embed_y_size)
        self.x_spatial_dim = x_spatial_dim

        self.embed_X = nn.Conv1d(embed_x_size, self.embed_size, 1)
        self.embed_Y = nn.Conv1d(embed_y_size, self.embed_size, 1)
        if x_spatial_dim is not None:
            self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)

            self.margin_X = nn.Conv1d(self.y_spatial_dim, 1, 1)
            self.margin_Y = nn.Conv1d(self.x_spatial_dim, 1, 1)

    def forward(self, X, Y=None):

        X_t = X.transpose(1, 2)
        Y_t = Y.transpose(1, 2) if Y is not None else X_t

        X_embed = self.embed_X(X_t)
        Y_embed = self.embed_Y(Y_t)

        X_norm = F.normalize(X_embed)
        Y_norm = F.normalize(Y_embed)

        S = X_norm.transpose(1, 2).bmm(Y_norm)
        if self.x_spatial_dim is not None:
            S = self.normalize_S(S.view(-1, self.x_spatial_dim * self.y_spatial_dim)) \
                .view(-1, self.x_spatial_dim, self.y_spatial_dim)

            X_poten = self.margin_X(S.transpose(1, 2)).transpose(1, 2).squeeze(2)
            Y_poten = self.margin_Y(S).transpose(1, 2).squeeze(2)
        else:
            X_poten = S.mean(dim=2, keepdim=False)
            Y_poten = S.mean(dim=1, keepdim=False)

        if Y is None:
            return X_poten
        else:
            return X_poten, Y_poten


class FactorGraphAttention(nn.Module):
    def __init__(self, util_e, high_order_utils=[], prior_flag=False,
                 sizes=[], size_flag=False, size_force=False, pairwise_flag=True, unary_flag=True, self_flag=True,
                 shared_weights=[], share_self=False, share_pairwise=True):
        """
        :param util_e:
        :param high_order_utils: list of tuples (util_index, num_of_utils, [connected_utils])
        :param prior_flag:
        :param sizes:
        :param size_flag:
        :param size_force:
        :param pairwise_flag:
        :param unary_flag:
        :param self_flag:
        """
        super(FactorGraphAttention, self).__init__()

        self.util_e = util_e

        self.prior_flag = prior_flag

        self.n_utils = len(util_e)

        self.spatial_pool = nn.ModuleDict()

        self.un_models = nn.ModuleList()

        self.self_flag = self_flag
        self.pairwise_flag = pairwise_flag
        self.unary_flag = unary_flag
        self.size_flag = size_flag
        self.size_force = size_force
        if not self.size_flag:
            sizes = [None for _ in util_e]
        self.high_order_utils = high_order_utils
        self.high_order_set = set([h[0] for h in self.high_order_utils])

        for idx, e_dim in enumerate(util_e):
            self.un_models.append(Unary(e_dim))
            if self.size_force:
                self.spatial_pool[str(idx)] = nn.AdaptiveAvgPool1d(sizes[idx])

        self.pp_models = nn.ModuleDict()
        self.shared_weights = shared_weights
        if len(shared_weights ) > 0:
            assert len(set([self.util_e[i] for i in shared_weights])) == 1, f"shared utils are of different dimensions " \
                                                                            f"{[self.util_e[i] for i in shared_weights]}"
            if share_pairwise:
                self.shared_pairwise = Pairwise(util_e[shared_weights[0]], None, util_e[shared_weights[0]], None)
            if share_self:
                self.shared_self = Pairwise(util_e[shared_weights[0]], None)
        for ((idx1, e_dim_1), (idx2, e_dim_2)) \
                in combinations_with_replacement(enumerate(util_e), 2):
            if idx1 == idx2:
                if idx1 in self.shared_weights and share_self:
                    self.pp_models[str(idx1)] = self.shared_self
                else:
                    self.pp_models[str(idx1)] = Pairwise(e_dim_1, sizes[idx1])
            else:
                if pairwise_flag:
                    for i, num_utils, connected_list in self.high_order_utils:
                        if i == idx1 and idx2 not in set(connected_list) \
                                or idx2 == i and idx1 not in set(connected_list):
                            continue
                    if idx1 in self.shared_weights and idx2 in shared_weights and share_pairwise:
                        self.pp_models[str((idx1, idx2))] = self.shared_pairwise
                    else:
                        self.pp_models[str((idx1, idx2))] = Pairwise(e_dim_1, sizes[idx1], e_dim_2, sizes[idx2])

        self.reduce_potentials = nn.ModuleList()
        self.num_of_potentials = dict()

        self.default_num_of_potentials = 0

        if self.self_flag:
            self.default_num_of_potentials += 1
        if self.unary_flag:
            self.default_num_of_potentials += 1
        if self.prior_flag:
            self.default_num_of_potentials += 1
        for idx in range(self.n_utils):
            self.num_of_potentials[idx] = self.default_num_of_potentials

        '''
        ' All other utils
        '''
        if pairwise_flag:
            for idx, num_utils, connected_utils in high_order_utils:
                for c_u in connected_utils:
                    self.num_of_potentials[c_u] += num_utils
                    self.num_of_potentials[idx] += 1
            for k in self.num_of_potentials.keys():
                if k not in self.high_order_set:
                    self.num_of_potentials[k] += (self.n_utils - 1) - len(high_order_utils)

        for idx in range(self.n_utils):
            self.reduce_potentials.append(nn.Conv1d(self.num_of_potentials[idx], 1, 1, bias=False))

    def forward(self, utils, priors=None, mode='full'):
        assert self.n_utils == len(utils)
        assert (priors is None and not self.prior_flag) \
               or (priors is not None
                   and self.prior_flag
                   and len(priors) == self.n_utils)
        b_size = utils[0].size(0)
        util_poten = dict()
        attention = list()
        if self.size_force:
            for i, num_utils, _ in self.high_order_utils:
                if str(i) not in self.spatial_pool.keys():
                    continue
                else:
                    high_util = utils[i]
                    high_util = high_util.view(num_utils * b_size, high_util.size(2), high_util.size(3))
                    high_util = high_util.transpose(1, 2)
                    utils[i] = self.spatial_pool[str(i)](high_util).transpose(1, 2)

            for i in range(self.n_utils):
                if i in self.high_order_set \
                        or str(i) not in self.spatial_pool.keys():
                    continue
                utils[i] = utils[i].transpose(1, 2)
                utils[i] = self.spatial_pool[str(i)](utils[i]).transpose(1, 2)
                if self.prior_flag and priors[i] is not None:
                    priors[i] = self.spatial_pool[str(i)](priors[i].unsqueeze(1)).squeeze(1)

        # local
        for i in range(self.n_utils):
            if i in self.high_order_set:
                continue
            if self.unary_flag:
                util_poten.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.self_flag:
                util_poten.setdefault(i, []).append(self.pp_models[str(i)](utils[i]))

        # joint
        if self.pairwise_flag:
            for (i, j) in combinations_with_replacement(range(self.n_utils), 2):
                if i in self.high_order_set \
                        or j in self.high_order_set:
                    continue
                if i == j:
                    continue
                else:
                    poten_ij, poten_ji = self.pp_models[str((i, j))](utils[i], utils[j])
                    util_poten.setdefault(i, []).append(poten_ij)
                    util_poten.setdefault(j, []).append(poten_ji)

        for i, num_utils, connected_list in self.high_order_utils:
            # i.e. High-Order utility
            high_util = utils[i]
            high_util = high_util.view(num_utils * b_size, high_util.size(2), high_util.size(3))
            if self.unary_flag:
                util_poten.setdefault(i, []).append(self.un_models[i](high_util))

            if self.self_flag:
                util_poten[i].append(self.pp_models[str(i)](high_util))

            if self.pairwise_flag:
                for j in connected_list:
                    other_util = utils[j]
                    if j in self.high_order_set:
                        expanded_util = other_util.view(b_size * num_utils, other_util.size(2), other_util.size(3))
                    else:
                        expanded_util = other_util.unsqueeze(1).expand(b_size,
                                                                       num_utils,
                                                                       other_util.size(1),
                                                                       other_util.size(2)).contiguous().view(
                            b_size * num_utils,
                            other_util.size(1),
                            other_util.size(2))
                    if j not in self.high_order_set:
                        if i < j:
                            poten_ij, poten_ji = self.pp_models[str((i, j))](high_util, expanded_util)
                        else:
                            poten_ji, poten_ij = self.pp_models[str((j, i))](expanded_util, high_util)
                        util_poten[i].append(poten_ij)
                        util_poten.setdefault(j, []).extend([u.squeeze(0) for u in
                                                             torch.split(poten_ji.view(b_size, num_utils, -1), 1, 1)])
                    else:
                        if i < j:
                            poten_ij, poten_ji = self.pp_models[str((i, j))](high_util, expanded_util)
                            util_poten[i].append(poten_ij)
                            util_poten.setdefault(j, []).extend([poten_ji])
        # utils
        for i in range(self.n_utils):
            if self.prior_flag:
                prior = priors[i] \
                    if priors[i] is not None \
                    else torch.zeros_like(util_poten[i][0], requires_grad=False).cuda()

                util_poten[i].append(prior)
            util_poten[i] = torch.cat([p if len(p.size()) == 3 else p.unsqueeze(1)
                                       for p in util_poten[i]], dim=1)
            util_poten[i] = self.reduce_potentials[i](util_poten[i]).squeeze(1)
            util_poten[i] = F.softmax(util_poten[i], dim=1).unsqueeze(2)
            if mode == 'full':
                if i in self.high_order_set:
                    high_order_util = utils[i].view(num_utils * b_size, utils[i].size(2), utils[i].size(3))
                    attention.append(torch.bmm(high_order_util.transpose(1, 2), util_poten[i]).view(b_size,
                                                                                                    num_utils, -1))
                else:
                    attention.append(torch.bmm(utils[i].transpose(1, 2), util_poten[i]).squeeze(2))

        return tuple(attention), util_poten

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
