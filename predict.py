# -*- coding:utf-8 -*-
# Author: Roger
# Created by Roger on 2017/10/24
from __future__ import absolute_import
import codecs
import sys, math
import numpy as np
import torch
from model import DocumentReaderQA
from corpus import WebQACorpus
import utils

def predict_answer(model, data_corpus, output_file=None, write_question=False, output_flag=False):
    answer_dict = dict()
    answer_dict_old = dict()
    correct_s, correct_e, correct = 0, 0, 0
    total = 0

    if output_flag:
        if output_file:
            output = codecs.open(output_file, 'w', 'utf8')
        else:
            output = sys.stdout
    else:
        output = None

    for question in data_corpus.next_question():
        q_key = str(question.q_keys[0])

        q_text = question.question_raw_text
        q_text = u''.join(q_text)

        pred_s, pred_e, pred_score, para_id = model.predict(question)

        # 计算每条(question, evidence)的准确率
        pred = np.stack((pred_s, pred_e))
        right_s = question.start_position.data.cpu().numpy()
        right_e = question.end_position.data.cpu().numpy()
        right = np.stack((right_s, right_e))
        compare = (pred == right)
        correct_se  = np.sum(compare, 1)
        correct_s += correct_se[0]
        correct_e += correct_se[1]
        correct += np.sum(np.sum(compare, 0) == 2)
        has_answer = np.sum(right_s >= 0)
#        total += len(pred_score)
        total += has_answer

        # 找出最大的score所对应的答案
        max_index = np.argmax(pred_score)
        start_position = pred_s[max_index]
        end_position = pred_e[max_index]
        evidence_id = para_id[max_index]
        answer_max = u''.join(question.evidence_raw_text[evidence_id][start_position:end_position + 1])
        answer_dict_old[q_key] = answer_max

        '''
        # 对于所有的evidence, 找出答案后 按score排序
        answers = []
        for i in range(len(pred_score)):
            start_position = pred_s[i]
            end_position = pred_e[i]
            evidence_id = para_id[i]
            answer = u''.join(question.evidence_raw_text[evidence_id][start_position:end_position + 1])
            answers.append(answer)
#        answers_sort = sorted(zip(answers, pred_score), key=lambda x:x[1], reverse=True)

        # 把相同的答案 分数合并
        answers_merge = {}
        cnt = {}
        for ans, score in zip(answers, pred_score):
            answers_merge[ans] = answers_merge.get(ans, 0) + math.sqrt(score)
            cnt[ans] = cnt.get(ans, 0) + 1
        for ans, score in answers_merge.items():
            answers_merge[ans] = score/math.sqrt(cnt[ans])
        answers_merge_sort = sorted(answers_merge.items(), key=lambda x:x[1], reverse=True)

        answer = answers_merge_sort[0][0]
        answer_dict[q_key] = answer

        # 输出
        if output_flag:
            if write_question:
                output.write("%s\t%s\t%s\n" % (q_key, q_text, answer))
            else:
                output.write("%s\t%s\n" % (q_key, answer))
        '''

    acc_s = correct_s / total
    acc_e = correct_e / total
    acc = correct / total

    return answer_dict_old, acc_s, acc_e, acc


def main():

    args = utils.add_argument()
    word_d, pos_d, ner_d = torch.load(args.dict_file)

    if args.device >= 0:
        model = torch.load(args.model_file, map_location=lambda storage, loc: storage.cuda(args.device))
    else:
        model = torch.load(args.model_file, map_location=lambda storage, loc: storage.cpu())

    model.eval()

    corpus = WebQACorpus(args.test_file, batch_size=args.batch, device=args.device,
                         word_dict=word_d, pos_dict=pos_d, ner_dict=ner_d)

    predict_answer(model, corpus, args.out_file, write_question=args.question, output_flag=True)


if __name__ == "__main__":
    main()
