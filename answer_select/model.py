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
        self.evidence_encoder = get_rnn(opt, self.embedding.output_size + 8)

        self.start_matcher = BilinearMatcher(self.evidence_encoder.output_size, self.question_encoder.output_size)
        self.end_matcher = BilinearMatcher(self.evidence_encoder.output_size, self.question_encoder.output_size)

        self.dropout = nn.Dropout(p=opt.dropout)

        self.ceLoss = nn.CrossEntropyLoss()

        self.device = opt.device

        self.reset_parameters()

    def answer_init(self, opt):
        self.rankingLoss = nn.MarginRankingLoss(opt.loss_margin)
        self.ones = Variable(torch.ones(opt.batch * opt.batch)).cuda(opt.device)

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

    def random_zeros(self, inputs, n):
        ones = torch.ones(inputs.size()).long()
        rand_idx = [np.random.choice(ones.size(0), n), np.random.choice(ones.size(1), n),
                    np.random.choice(ones.size(2), n)]
        ones[rand_idx] = 0
        return inputs*Variable(ones).cuda(self.device)

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
#        e_input = torch.cat([batch.e_text.unsqueeze(-1), batch.e_feature[:, :, :2]], dim=-1)
        e_input = batch.e_text
        e_word_emb = self.embedding.forward(e_input)

        evidence_input_emb = [e_word_emb, batch.e_feature]

        if aligned_feature is not None:
            evidence_input_emb.append(aligned_feature)

        evidence_input = torch.cat(evidence_input_emb, dim=2)

        evidence_hidden, _ = self.evidence_encoder.forward(evidence_input, lengths=batch.e_lens)

        return evidence_hidden

    def score(self, batch):
        # (batch, q_size)
        question_embedding = self.get_question_embedding(batch)

#        q_word_emb = self.embedding.forward(batch.q_text)
#        e_word_emb = self.embedding.forward(batch.e_text)
#        aligned_feature = self.get_soft_align_embedding(q_word_emb, e_word_emb, batch.q_lens, batch.e_lens)

        # (batch, e_len, e_size)
#        evidence_embedding = self.get_evidence_embedding(batch, aligned_feature)
        evidence_embedding = self.get_evidence_embedding(batch)

        # Size Check
        q_batch_size, q_emb_size = question_embedding.size()
        batch_size, e_max_len, e_emb_size = evidence_embedding.size()
        if batch_size != q_batch_size:
            assert batch_size == q_batch_size*2
            question_embedding = question_embedding.unsqueeze(0).expand(2, q_batch_size,
                                                   q_emb_size).contiguous().view(batch_size,q_emb_size)

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

    def loss_max(self, batch, max_len=15):
        # (batch, e_len)
        start_score, end_score = self.score(batch)
        pos_size = start_score.size(0)//2

        pos_index = self.pos_index[:pos_size]
        neg_index = self.neg_index[:pos_size]
        ones = self.ones[:pos_size]

        start_right_score = torch.index_select(batch.start_position, 0, pos_index)
        end_right_score = torch.index_select(batch.end_position, 0, pos_index)

        pos_start_score = torch.index_select(start_score, 0, pos_index)
        pos_end_score = torch.index_select(end_score, 0, pos_index)

        start_loss = self.ceLoss(pos_start_score, start_right_score)
        end_loss = self.ceLoss(pos_end_score, end_right_score)

        # (batch, e_len, e_len)
        scores = start_score.unsqueeze(2) + end_score.unsqueeze(1)
        for i in range(scores.size(0)):
            scores[i] = torch.tril(torch.triu(scores[i]), max_len - 1)

        # (batch, )
        pred_score = torch.max(torch.max(scores, 2)[0], 1)[0]
        pos_score = torch.index_select(pred_score, 0, pos_index)
        neg_score = torch.index_select(pred_score, 0, neg_index)

        ranking_loss = self.rankingLoss(pos_score, neg_score, ones)

        loss = start_loss + end_loss + ranking_loss
        return loss

    def loss(self, batch):
        # (batch, e_len)
        start_score, end_score = self.score(batch)
        pos_size = start_score.size(0)//2

        start_right_score = batch.start_position[:pos_size]
        end_right_score = batch.end_position[:pos_size]

        pos_start_score = start_score[:pos_size]
        pos_end_score = end_score[:pos_size]
        neg_start_score = start_score[pos_size:]
        neg_end_score = end_score[pos_size:]

        start_loss = self.ceLoss(pos_start_score, start_right_score)
        end_loss = self.ceLoss(pos_end_score, end_right_score)

        # (batch, )
        pos_start_max, ps_m = torch.max(pos_start_score, 1)
        neg_start_max, _ = torch.max(neg_start_score, 1)
        pos_start_max = torch.masked_select(pos_start_max, ps_m==start_right_score)
        neg_start_max = torch.masked_select(neg_start_max, ps_m==start_right_score)
        pos_end_max, pe_m = torch.max(pos_end_score, 1)
        neg_end_max, _ = torch.max(neg_end_score, 1)
        pos_end_max = torch.masked_select(pos_end_max, pe_m==end_right_score)
        neg_end_max = torch.masked_select(neg_end_max, pe_m==end_right_score)

        ones_s = self.ones[:pos_start_max.size(0)]
        ones_e = self.ones[:pos_end_max.size(0)]
        ranking_start_loss = self.rankingLoss(pos_start_max, neg_start_max, ones_s)
        ranking_end_loss = self.rankingLoss(pos_end_max, neg_end_max, ones_e)

        loss = start_loss + end_loss + ranking_start_loss + ranking_end_loss
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
#            scores = torch.ger(score_s[i], score_e[i])
            scores = score_s[i].unsqueeze(1) + score_e[i].unsqueeze(0)
#            scores = torch.exp(score_s[i]).unsqueeze(1) + torch.exp(score_e[i]).unsqueeze(0)

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
