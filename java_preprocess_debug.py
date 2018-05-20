import torch
from corpus33 import WebQACorpus
import utils

args = utils.add_argument()
word_d, pos_d, ner_d = torch.load(args.dict_file)
corpus = WebQACorpus('data/baidu_data.json', batch_size=args.batch, device=args.device,
                         word_dict=word_d, pos_dict=pos_d, ner_dict=ner_d)

for question in corpus.next_question():
    q_key, q_text, e_key, e_text = question.q_keys[0], question.question_raw_text, question.e_keys, question.evidence_raw_text
    start, end = question.start_position, question.end_position
    print(q_key, u' '.join(q_text))
    for i in range(len(e_key)):
        print(e_key[i], u' '.join(e_text[i]))
#        if len(start[i]) != len(end[i]):
#            print(q_key, e_key[i])
        print(start[i])
        print(end[i])
