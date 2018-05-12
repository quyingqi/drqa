# -*- coding:utf-8 -*-
# Author: Roger
# Created by Roger on 2017/10/24
from __future__ import absolute_import
import codecs
import sys
import numpy as np
import torch
from model import DocumentReaderQA
from corpus import WebQACorpus
import utils

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

        pred_s, pred_e, pred_score, para_id = model.predict(question)

        max_index = np.argmax(pred_score)
        start_position = pred_s[max_index][0]
        end_position = pred_e[max_index][0]
        evidence_id = para_id[max_index]

        answer = u''.join(question.evidence_raw_text[evidence_id][start_position:end_position + 1])
        q_text = u''.join(q_text)

        answer_dict[q_key] = answer
        if output_flag:
            if write_question:
                output.write("%s\t%s\t%s\n" % (q_key, q_text, answer))
            else:
                output.write("%s\t%s\n" % (q_key, answer))

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
