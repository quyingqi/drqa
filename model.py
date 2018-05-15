# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/28
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from layers.mask_util import lengths2mask
from layers import Embeddings, BilinearMatcher, DotWordSeqAttetnion, PaddBasedRNNEncoder, RNNEncoder


def get_rnn(opt, input_size):
    if opt.multi_layer_hidden == 'concatenate':
        rnn = PaddBasedRNNEncoder(input_size=input_size,
                                  hidden_size=opt.hidden_size,
                                  num_layers=opt.num_layers,
                                  dropout=opt.encoder_dropout,
                                  brnn=opt.brnn,
                                  rnn_type=opt.rnn_type,
                                  multi_layer_hidden='concatenate')
    elif opt.multi_layer_hidden == 'last':
        rnn = RNNEncoder(input_size=input_size,
                                  hidden_size=opt.hidden_size,
                                  num_layers=opt.num_layers,
                                  dropout=opt.encoder_dropout,
                                  brnn=opt.brnn,
                                  rnn_type=opt.rnn_type,
                                  multi_layer_hidden='last')
    else:
        raise NotImplementedError
    return rnn


class DocumentReaderQA(nn.Module):
    def __init__(self, dicts, opt, feat_dicts, feat_dims):
        super(DocumentReaderQA, self).__init__()

        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts,
                                    feature_dicts=feat_dicts,
                                    feature_dims=feat_dims)

        self.question_encoder = get_rnn(opt, self.embedding.output_size)

        self.question_attention_p = nn.Parameter(torch.Tensor(self.question_encoder.output_size))

        self.question_attention = DotWordSeqAttetnion(input_size=self.question_encoder.output_size,
                                                      seq_size=self.question_encoder.output_size)

#        self.soft_align_linear = nn.Linear(self.embedding.output_size, self.embedding.output_size)
        self.soft_align_linear = nn.Linear(opt.word_vec_size, opt.word_vec_size)

#        self.evidence_encoder = get_rnn(opt, self.embedding.output_size + 1 + opt.word_vec_size)
        self.evidence_encoder = get_rnn(opt, self.embedding.output_size + 1)

        self.start_matcher = BilinearMatcher(self.evidence_encoder.output_size, self.question_encoder.output_size)
        self.end_matcher = BilinearMatcher(self.evidence_encoder.output_size, self.question_encoder.output_size)

        self.dropout = nn.Dropout(p=opt.dropout)

        self.ceLoss = nn.CrossEntropyLoss()

        self.device = opt.device

        self.reset_parameters()

    def reset_parameters(self):
        self.question_attention_p.data.normal_(0, 1)

    def get_soft_align_embedding(self, q_word_emb, e_word_emb, q_lens, e_lens):
        # (batch, q_len, word_size)
        # (batch, e_len, word_size)
        if self.soft_align_linear:
            q_word_proj = F.relu(self.soft_align_linear(q_word_emb))
            e_word_proj = F.relu(self.soft_align_linear(e_word_emb))
        else:
            q_word_proj = q_word_emb
            e_word_proj = e_word_emb

        batch_size_q, q_maxlen, word_size_q = q_word_proj.size()
        batch_size_e, e_maxlen, word_size_e = e_word_proj.size()
        assert batch_size_q == batch_size_e
        assert word_size_q == word_size_e

        # (batch, e_len, word_size) dot (batch, q_len, word_size) -> (batch, e_len, q_len)
        scores = torch.bmm(e_word_proj, q_word_proj.transpose(2, 1))

        # (batch, e_len) -> (batch, e_len, q_len)
        # (batch, q_len) -> (batch, e_len, q_len)
        e_mask = lengths2mask(e_lens, e_maxlen, byte=True).unsqueeze(-1).expand(scores.size())
        q_mask = lengths2mask(q_lens, q_maxlen, byte=True).unsqueeze(1).expand(scores.size())
        e_mask = 1 + e_mask * -1
        q_mask = 1 + q_mask * -1

        scores = scores.data.masked_fill_(q_mask.data, float("-inf"))
        batch, e_len, q_len = scores.size()
        scores = scores.view(batch * e_len, q_len)
        # (batch, e_len, q_len, 1)
        weight = F.softmax(scores).view(batch, e_len, q_len)
        weight = weight.masked_fill_(e_mask, 0).unsqueeze(-1)

        # (batch, q_len, word_size) -> (batch, e_len, q_len, word_size)
        q_word_emb_expand = q_word_emb.unsqueeze(1).expand(batch, e_len, q_len, q_word_emb.size(2))

        # (batch, e_len, word_size)
        e_emb_aligned = torch.sum(weight * q_word_emb_expand, 2)
        return e_emb_aligned

    def get_question_embedding(self, batch):
        question_attention_p = self.question_attention_p.unsqueeze(0)
        question_attention_p = question_attention_p.expand(batch.batch_size, self.question_attention_p.size(0))

        q_input = torch.cat([batch.q_text.unsqueeze(-1), batch.q_feature], dim=-1)
        q_word_emb = self.embedding.forward(q_input)

        q_hidden_embs, _ = self.question_encoder.forward(q_word_emb, lengths=batch.q_lens)
        q_hidden_embs = q_hidden_embs.contiguous()

        q_hidden_emb, _ = self.question_attention.forward(question_attention_p, q_hidden_embs, lengths=batch.q_lens)

        return q_hidden_emb

    def get_evidence_embedding(self, batch, aligned_feature=None):
        e_input = torch.cat([batch.e_text.unsqueeze(-1), batch.e_feature[:, :, :2]], dim=-1)
        e_word_emb = self.embedding.forward(e_input)

        evidence_input_emb = [e_word_emb, batch.e_feature[:, :, -1].unsqueeze(-1).float()]

        if aligned_feature is not None:
            evidence_input_emb.append(aligned_feature)

        evidence_input = torch.cat(evidence_input_emb, dim=2)

        evidence_hidden, _ = self.evidence_encoder.forward(evidence_input, lengths=batch.e_lens)

        return evidence_hidden

    def score(self, batch):
        # (batch, q_size)
        question_embedding = self.get_question_embedding(batch)

        q_word_emb = self.embedding.forward(batch.q_text)
        e_word_emb = self.embedding.forward(batch.e_text)
        aligned_feature = self.get_soft_align_embedding(q_word_emb, e_word_emb, batch.q_lens, batch.e_lens)
        # (batch, e_len, e_size)
#        evidence_embedding = self.get_evidence_embedding(batch, aligned_feature)
        evidence_embedding = self.get_evidence_embedding(batch)

        # Size Check
        batch_size = question_embedding.size(0)
        q_emb_size = question_embedding.size(1)

        e_max_len = evidence_embedding.size(1)
        e_emb_size = evidence_embedding.size(2)

        assert batch_size == evidence_embedding.size(0)

        # (batch, e_len)
        e_mask = lengths2mask(batch.e_lens, e_max_len)

        # (batch, q_size) -> # (batch, 1, q_size)
        q_embedding = question_embedding.unsqueeze(1)

        # (batch, 1, q_size) -> # (batch, e_len, q_size)
        question_embedding = q_embedding.expand(batch_size, e_max_len, q_emb_size)

        question_embedding = question_embedding.contiguous()

        # (batch, e_len, q_size) -> # (batch * e_len, q_size)
        question_embedding = question_embedding.view(batch_size * e_max_len, q_emb_size)

        e_embedding = evidence_embedding.contiguous()
        # (batch, e_len, e_size) -> # (batch * e_len, e_size)
        evidence_embedding = e_embedding.view(batch_size * e_max_len, e_emb_size)

        # (batch * e_len, q_size) (batch * e_len, e_size) -> batch * e_len
        start_score = self.start_matcher.forward(evidence_embedding, question_embedding).squeeze(-1)
        end_score = self.end_matcher.forward(evidence_embedding, question_embedding).squeeze(-1)

        # batch * e_len -> (batch, e_len)
        start_score = start_score.view(batch_size, e_max_len) * e_mask
        end_score = end_score.view(batch_size, e_max_len) * e_mask

        '''
        # sim_dot
        q_embedding = question_embedding.unsqueeze(1)
        e_embedding = evidence_embedding.contiguous()
        start_score = torch.bmm(q_embedding, e_embedding.transpose(1, 2)).squeeze(1) * e_mask
        end_score = torch.bmm(q_embedding, e_embedding.transpose(1, 2)).squeeze(1) * e_mask
        '''

        return start_score, end_score

    def loss_old(self, batch):
        start_score, end_score = self.score(batch)

        def log_sum_exp(x, dim=0):
            """
            :param x: (batch, label)
            :return:
            """
            max_value, _ = torch.max(x, dim)
            max_exp = max_value.unsqueeze(dim).expand_as(x)
            return max_value + torch.log(torch.sum(torch.exp(x - max_exp), dim))

        start_norm = log_sum_exp(start_score, 1)
        end_norm = log_sum_exp(end_score, 1)

        # (batch, len) -> (batch, 1)
        start_right_score = torch.gather(start_score, 1, batch.start_position.unsqueeze(-1))
        end_right_score = torch.gather(end_score, 1, batch.end_position.unsqueeze(-1))

        start_loss = torch.mean(start_norm - start_right_score)
        end_loss = torch.mean(end_norm - end_right_score)

        loss = start_loss + end_loss

        return loss

    def loss(self, batch):
        start_score, end_score = self.score(batch)

        start_right_score = batch.start_position
        end_right_score = batch.end_position

        start_loss = self.ceLoss(start_score, start_right_score)
        end_loss = self.ceLoss(end_score, end_right_score)

        loss = start_loss + end_loss
        return loss

    def loss_3(self, batch):
        start_score, end_score = self.score(batch)

        _, start_pos = torch.max(start_score, dim=1)
        _, end_pos = torch.max(end_score, dim=1)
        start_pos = start_pos.data.cpu()
        end_pos = end_pos.data.cpu()

        right_s = batch.start_position
        right_e = batch.end_position

        best_start = [x[0] for x in right_s]
        best_end = [x[0] for x in right_e]

        for i in range(len(best_start)):
            if len(right_s[i]) == len(right_e[i]) and start_pos[i] in right_s[i]:
                best_start[i] = start_pos[i]
                best_end[i] = end_pos[i]

        start_right_score = Variable(torch.LongTensor(best_start).cuda(self.device))
        end_right_score = Variable(torch.LongTensor(best_end).cuda(self.device))

        start_loss = self.ceLoss(start_score, start_right_score)
        end_loss = self.ceLoss(end_score, end_right_score)

        loss = start_loss + end_loss
        return loss

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.
        from https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/model.py
        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        para_id = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            if isinstance(scores, Variable):
                scores = scores.data

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.cpu().numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)

            pred_s.append(s_idx[0])  # 默认取top1，否则 改成s_idx
            pred_e.append(e_idx[0])
            pred_score.append(scores_flat[idx_sort])
            para_id.append(i)
        return pred_s, pred_e, pred_score, para_id

    def predict(self, q_evidens, top_n=1, max_len=15):
        # (batch, e_len)
        start_score, end_score = self.score(q_evidens)

        pred_s, pred_e, pred_score, para_id = self.decode(start_score, end_score, top_n=top_n, max_len=max_len)

        return pred_s, pred_e, pred_score, para_id

    def forward(self, batch):
        return self.loss(batch)
