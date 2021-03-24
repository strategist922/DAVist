from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import logging
import time
import numpy as np
from .model_utils import AttentionLayer, VisualEncoder, _smallest, weight_init
from misc.utils import create_occurance_matrix


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.story_size = opt.story_size
        self.word_embed_dim = opt.word_embed_dim
        self.hidden_dim = opt.hidden_dim
        self.num_layers = opt.num_layers
        self.num_layers_decoder = opt.num_layers_decoder
        self.rnn_type = opt.rnn_type
        self.dropout = opt.dropout
        self.seq_length = opt.seq_length
        self.feat_size = opt.feat_size
        self.decoder_input_dim = self.word_embed_dim + self.word_embed_dim
        self.ss_prob = 0.0  # Schedule sampling probability

        # Visual Encoder
        self.encoder = VisualEncoder(opt)


        # Decoder LSTM
        self.project_d = nn.Linear(self.decoder_input_dim, self.word_embed_dim)
        if self.rnn_type == 'gru':
            self.decoder = nn.GRU(input_size=self.word_embed_dim, hidden_size=self.hidden_dim, batch_first=True,
                                  num_layers=self.num_layers_decoder)
        elif self.rnn_type == 'lstm':
            self.decoder = nn.LSTM(input_size=self.word_embed_dim, hidden_size=self.hidden_dim, batch_first=True,
                                   num_layers=self.num_layers_decoder)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # last linear layer
        self.logit = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                   nn.Tanh(),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(self.hidden_dim // 2, self.vocab_size))
        self.init_s_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.init_c_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # self.project_d_gate = nn.Linear(self.hidden_dim, 32)
        # self.project_h_gate = nn.Linear(32, 32)
        # self.gate = nn.Linear(32, 1)

        # self.count_prior_h = nn.Sequential(nn.Linear(self.vocab_size, 32),
        #                                  nn.Dropout(0.1))
        # self.count_prior = nn.Sequential(nn.Linear(32, self.vocab_size),
        #                                  nn.ReLU())

        for m in self.parameters():
            weight_init(m)
            
        # word embedding layer
        if opt.from_pretrained:
            self.embed = nn.Embedding.from_pretrained(torch.load(opt.embedding_file), freeze=False)
        else:
            self.embed = nn.Embedding(self.vocab_size, self.word_embed_dim)
            weight_init(self.embed.parameters())

    def init_weights(self, init_range):
        logging.info("Initialize the parameters of the model")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return weight.new(self.num_layers_decoder * times, batch_size, dim).zero_()
        else:
            return (weight.new(self.num_layers_decoder * times, batch_size, dim).zero_(),
                    weight.new(self.num_layers_decoder * times, batch_size, dim).zero_())

    def init_hidden_with_feature(self, feature):
        if self.rnn_type == 'gru':
            output = self.init_s_proj(feature)
            return output.view(1, -1, output.size(-1)).expand(self.num_layers, -1, output.size(-1)).contiguous()
        else:
            output1 = self.init_s_proj(feature)
            output2 = self.init_c_proj(feature)
            return (output1.view(1, -1, output1.size(-1)).expand(self.num_layers, -1, output1.size(-1)).contiguous(), \
                    output2.view(1, -1, output2.size(-1)).expand(self.num_layers, -1, output2.size(-1)).contiguous())


    def decode(self, imgs, last_word, state_d, history_count=None, penalize_previous=False):
        # 'last_word' is Variable contraining a word index
        # batch_size * input_encoding_size
        word_emb = self.embed(last_word)
        word_emb = torch.unsqueeze(word_emb, 1)

        input_d = torch.cat([word_emb, imgs.unsqueeze(1)], 2)  # batch_size * 1 * dim
        input_d = self.project_d(input_d)

        out_d, state_d = self.decoder(input_d, state_d)

        log_probs = F.log_softmax(self.logit(out_d[:, 0, :]))

        if penalize_previous:
            last_word_onehot = torch.FloatTensor(last_word.size(0), self.vocab_size).zero_().cuda()
            penalize_value = (last_word > 0).data.float() * -100
            mask = Variable(last_word_onehot.scatter_(1, last_word.data[:, None], 1.) * penalize_value[:, None])
            log_probs = log_probs + mask

        return log_probs, state_d

    def forward(self, features_fc, features_obj, caption, history_count, frequencies=None, spatial=None, clss=None, attrs=None):
        """
        :param features_fc: (batch_size, 5, feat_size)
        :param features_obj: (batch_size, 5, num_obj, feat_size)
        :param caption: (batch_size, 5, seq_length)
        :param spatial: (batch_size, 5, num_obj, num_spatial)
        :param clss: (batch_size, 5, num_obj)
        :param attrs: (batch_size, 5, num_obj)
        :return:
        """
        # encode the visual features
        out_e, _ = self.encoder(features_fc, features_obj, spatial=spatial, clss=clss, attrs=attrs)

        # initialize decoder's state
        # state_d = self.init_hidden(batch_size, bi=False, dim=self.hidden_dim)
        state_d = self.init_hidden_with_feature(out_e)

        # reshape the inputs, making the sentence generation separately
        out_e = out_e.view(-1, out_e.size(2))
        caption = caption.view(-1, caption.size(2))

        ############################# decoding stage ##############################
        batch_size = out_e.size(0)


        last_word = torch.FloatTensor(batch_size).long().zero_().cuda()
        outputs = []
        # history_count_numpy = history_count.view(-1, self.vocab_size).cpu().numpy()
        # if frequencies is not None:
        #     history_count_numpy = np.maximum(history_count_numpy - frequencies + 1, np.zeros_like(history_count_numpy))
        for i in range(self.seq_length):
            log_probs, state_d = self.decode(out_e, last_word, state_d)
            outputs.append(log_probs)

            # choose the word
            if self.ss_prob > 0.0:
                sample_prob = torch.FloatTensor(batch_size).uniform_(0, 1).cuda()
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    last_word = caption[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    last_word = caption[:, i].data.clone()
                    # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(log_probs.data)
                    last_word.index_copy_(0, sample_ind,
                                          torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    last_word = Variable(last_word)
            else:
                last_word = caption[:, i].clone()

            # break condition
            if i >= 1 and caption[:, i].data.sum() == 0:
                break

        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)  # batch_size * 5, -1, vocab_size
        return outputs.view(-1, self.story_size, outputs.size(1), self.vocab_size)

    def sample(self, features_fc, features_obj, sample_max, penalty=0, rl_training=False, pad=False, spatial=None,
               clss=None, attrs=None, frequencies=None, function_words=[]):
        # encode the visual features
        out_e, _ = self.encoder(features_fc, features_obj, spatial=spatial, clss=clss, attrs=attrs)
        # reshape the inputs, making the sentence generation separately
        # out_e = out_e.view(-1, out_e.size(2))

        ###################### Decoding stage ###############################
        batch_size = out_e.size(0)

        seqs = []
        seqs_log_probs = []
        # counter = np.zeros((batch_size, self.vocab_size), 'float32')
        baselines = []
        for i in range(self.story_size):
            state_d = self.init_hidden_with_feature(out_e[:, i, :].unsqueeze(1))
            out_e_i = out_e[:, i, :]
            seq = []
            seq_log_probs = []
            if rl_training:
                baseline = []

            last_word = torch.FloatTensor(batch_size).long().zero_().cuda()
            for t in range(self.seq_length):
                log_probs, state_d = self.decode(out_e_i, last_word, state_d, None,
                                                 True)
                log_probs[:, 1] = log_probs[:, 1] - 1000
                if t < 6:
                    mask = np.zeros((batch_size, log_probs.size(-1)), 'float32')
                    mask[:, 0] = -1000
                    mask = torch.from_numpy(mask).cuda()
                    log_probs = log_probs + mask
                if sample_max:
                    if penalty > 0:
                        sample_log_prob = torch.exp(log_probs.data)
                        # curr_counter = counter.copy()
                        if frequencies is not None:
                            curr_counter = np.maximum(curr_counter - frequencies + 1, np.zeros_like(curr_counter))
                        sample_log_prob = sample_log_prob/(penalty*torch.from_numpy(curr_counter).cuda() + 1)
                        sample_log_prob = sample_log_prob/torch.sum(sample_log_prob, 1).unsqueeze(1)
                        sample_log_prob, last_word = torch.max(sample_log_prob, 1)
                        last_word = last_word.data.view(-1).long()
                    else:
                        sample_log_prob, last_word = torch.max(log_probs, 1)
                        last_word = last_word.data.view(-1).long()
                else:
                    # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(log_probs.data).cpu()
                    last_word = torch.multinomial(prob_prev, 1).cuda()
                    # gather the logprobs at sampled positions
                    sample_log_prob = log_probs.gather(1, last_word)
                    # flatten indices for downstream processing
                    last_word = last_word.view(-1).long()
                # counter[range(counter.shape[0]), last_word.cpu().numpy()] += 1
                # counter[:, function_words] = 0

                if t == 0:
                    unfinished = last_word > 0
                else:
                    unfinished = unfinished * (last_word > 0)
                if unfinished.sum() == 0 and t >= 1 and not pad:
                    break
                last_word = last_word * unfinished.type_as(last_word)

                seq.append(last_word)  # seq[t] the input of t time step
                seq_log_probs.append(sample_log_prob.view(-1))
                if rl_training:
                    # cut off the gradient passing using detech()
                    value = self.baseline_estimator(state_d[0].detach())
                    baseline.append(value)

            seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)  # batch_size * 5, seq_length
            seq_log_probs = torch.cat([_.unsqueeze(1) for _ in seq_log_probs], 1)
            seqs.append(seq)
            seqs_log_probs.append(seq_log_probs)
            if rl_training:
                baseline = torch.cat([_.unsqueeze(1) for _ in baseline], 1)  # batch_size * 5, seq_length
                baseline = baseline.view(-1, self.story_size, baseline.size(1))
                baselines.append(baseline)

        seq_max = max([s.size(1) for s in seqs])
        seqs = [F.pad(s, (0, seq_max - s.size(1))) for s in seqs]
        seqs = torch.cat(seqs, 1).view(-1, self.story_size, seq_max)
        seqs_log_probs = [F.pad(s, (0, seq_max - s.size(1))) for s in seqs_log_probs]
        seqs_log_probs = torch.cat(seqs_log_probs, 1).view(-1, self.story_size, seq_max)
        return seqs, seqs_log_probs


    def predict(self, features_fc, features_obj, beam_size=5, penalty=0, spatial=None, clss=None, attrs=None,
                frequencies=None, function_words=[]):
        assert beam_size <= self.vocab_size and beam_size > 0
        if beam_size == 1:  # if beam_size is 1, then do greedy decoding, otherwise use beam search
            return self.sample(features_fc, features_obj, sample_max=True, penalty=penalty, spatial=spatial,
                               clss=clss, attrs=attrs, frequencies=frequencies, function_words=function_words)

        # encode the visual features
        out_e, _ = self.encoder(features_fc, features_obj, spatial=spatial, clss=clss, attrs=attrs)
        # reshape the inputs, making the sentence generation separately
        # out_e = out_e.view(-1, out_e.size(2))

        ####################### decoding stage ##################################
        batch_size = out_e.size(0)
        # initialize decoder's state
        # state_d = self.init_hidden(batch_size, bi=False, dim=self.hidden_dim)

        seqs = []
        seqs_log_probs = []
        # lets process the videos independently for now, for simplicity
        counter = np.zeros((batch_size, self.vocab_size), 'float32')
        for j in range(self.story_size):
            seq = torch.LongTensor(self.seq_length, batch_size).zero_()
            seq_log_probs = torch.FloatTensor(self.seq_length, batch_size)
            state_d = self.init_hidden_with_feature(out_e[:, j, :].unsqueeze(1))
            out_e_j = out_e[:, j, :]
            for k in range(batch_size):
                # curr_counter = np.zeros((beam_size, self.vocab_size), 'float32')
                out_e_k = out_e_j[k].unsqueeze(0).expand(beam_size, out_e_j.size(1))
                if self.rnn_type != 'lstm':
                    state_d_k = state_d[:, k, :].unsqueeze(1).expand(state_d.size(0), beam_size,
                                                                     state_d.size(2)).contiguous()
                else:
                    state_d_k = tuple(
                        [state_di[:, k, :].unsqueeze(1).expand(state_di.size(0), beam_size, state_di.size(2)).contiguous()
                         for state_di in state_d])
                last_word = torch.FloatTensor(beam_size).long().zero_().cuda()  # <BOS>

                log_probs, state_d_k = self.decode(out_e_k, last_word, state_d_k, None,
                                                   True)
                log_probs[:, 1] = log_probs[:, 1] - 1000  # never produce <UNK> token

                neg_log_probs = -log_probs

                all_outputs = np.ones((1, beam_size), dtype='int32')
                all_masks = np.ones_like(all_outputs, dtype="float32")
                all_costs = np.zeros_like(all_outputs, dtype="float32")
                for i in range(self.seq_length):
                    if all_masks[-1].sum() == 0:
                        break

                    next_costs = (all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None])
                    (finished,) = np.where(all_masks[-1] == 0)
                    next_costs[finished, 1:] = np.inf

                    (indexes, outputs), chosen_costs = _smallest(next_costs, beam_size, only_first_row=i == 0)
                    if self.rnn_type != 'lstm':
                        new_state_d = state_d_k.data.cpu().numpy()[:, indexes, :]
                    else:
                        new_state_d = tuple([state_d_ki.data.cpu().numpy()[:, indexes, :] for state_d_ki in state_d_k])

                    all_outputs = all_outputs[:, indexes]
                    all_masks = all_masks[:, indexes]
                    all_costs = all_costs[:, indexes]
                    # curr_counter = curr_counter[indexes, :]


                    last_word = torch.from_numpy(outputs).cuda()
                    if self.rnn_type != 'lstm':
                        state_d_k = torch.from_numpy(new_state_d).cuda()
                    else:
                        state_d_k = tuple([torch.from_numpy(new_state_di).cuda() for new_state_di in new_state_d])

                    log_probs, state_d_k = self.decode(out_e_k, last_word, state_d_k, None, True)

                    log_probs[:, 1] = log_probs[:, 1] - 1000
                    if penalty > 0:
                        sample_log_prob = torch.exp(log_probs.data)
                        curr_counter_i = curr_counter + counter[k]
                        if frequencies is not None:
                            curr_counter_i = curr_counter_i - frequencies + 1
                            curr_counter_i[curr_counter_i < 0] = 0.0
                        sample_log_prob = sample_log_prob / (
                                    penalty * torch.from_numpy(curr_counter_i).cuda() + 1)
                        sample_log_prob = sample_log_prob / torch.sum(sample_log_prob, 1).unsqueeze(1)
                        log_probs = torch.log(sample_log_prob)

                    neg_log_probs = -log_probs

                    all_outputs = np.vstack([all_outputs, outputs[None, :]])
                    all_costs = np.vstack([all_costs, chosen_costs[None, :]])
                    mask = outputs != 0
                    all_masks = np.vstack([all_masks, mask[None, :]])
                    # curr_counter = create_occurance_matrix(all_outputs.T, self.vocab_size)
                    # curr_counter[:, function_words] = 0

                all_outputs = all_outputs[1:]
                all_costs = all_costs[1:] - all_costs[:-1]
                all_masks = all_masks[:-1]
                costs = all_costs.sum(axis=0)
                lengths = all_masks.sum(axis=0)
                normalized_cost = costs / lengths
                best_idx = np.argmin(normalized_cost)
                seq[:all_outputs.shape[0], k] = torch.from_numpy(all_outputs[:, best_idx])
                seq_log_probs[:all_costs.shape[0], k] = torch.from_numpy(all_costs[:, best_idx])
                # counter[k] += curr_counter[best_idx]

            # return the samples and their log likelihoods
            seq = seq.transpose(0, 1)
            seq_log_probs = seq_log_probs.transpose(0, 1)
            seqs.append(seq)
            seqs_log_probs.append(seq_log_probs)
        seqs = torch.cat(seqs, 1).view(-1, self.story_size, self.seq_length)
        seqs_log_probs = torch.cat(seqs_log_probs, 1).view(-1, self.story_size, self.seq_length)
        return seqs, seqs_log_probs
