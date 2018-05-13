# -*- coding:utf-8 -*-
# Author: Roger
# Created by Roger on 2017/10/24
from __future__ import absolute_import
import codecs
import sys, os, math
import numpy as np
import torch
from model import DocumentReaderQA
from corpus import WebQACorpus
import evaluate
import utils

qid_answer_expand = evaluate.load_qid_answer_expand('data/qid_answer_expand/qid_answer_expand.all')
def predict_answer(model, data_corpus, output_file=None, write_question=False, output_flag=False):
    answer_dict = dict()

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

        # 找出最大的score所对应的答案
        max_index = np.argmax(pred_score)
        start_position = pred_s[max_index][0]
        end_position = pred_e[max_index][0]
        evidence_id = para_id[max_index]
        answer_max = u''.join(question.evidence_raw_text[evidence_id][start_position:end_position + 1])

        # 对于所有的evidence, 找出答案后 按score排序
        answers = []
        for i in range(len(pred_score)):
            start_position = pred_s[i][0]
            end_position = pred_e[i][0]
            evidence_id = para_id[i]
            answer = u''.join(question.evidence_raw_text[evidence_id][start_position:end_position + 1])
            answers.append(answer)
        answers_sort = sorted(zip(answers, pred_score), key=lambda x:x[1], reverse=True)

        # 把相同的答案 分数合并
        answers_merge = {}
        for ans, score in answers_sort:
            answers_merge[ans] = answers_merge.get(ans, 0) + math.log(score+1)
        answers_merge_sort = sorted(answers_merge.items(), key=lambda x:x[1], reverse=True)
        gold = qid_answer_expand[q_key][1]

        answer = answers_merge_sort[0][0]
        answer_dict[q_key] = answer

        # 输出到日志文件
        if output_flag:
            if write_question:
                is_match = evaluate.is_exact_match_answer(q_key, answer, qid_answer_expand)
                gold = qid_answer_expand[q_key][1]
                output.write("%s\t%s\t%s\t%s\t%s\n" % (q_key, q_text, gold, answers_merge_sort, is_match))
            else:
                output.write("%s\t%s\n" % (q_key, answer))

    q_level_p, char_level_f = evaluate.evalutate(answer_dict)
    print('q_level_p: %.2f\tchar_level_f: %.2f' %(q_level_p, char_level_f))
    return answer_dict


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
