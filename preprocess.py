#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/28
from __future__ import absolute_import
import torch
import argparse
from utils import get_data_dict, add_argument
from corpus import WebQACorpus


def preprocess_data(args):
    baidu_file = 'data/baidu_data.json'
    train_file = 'data/sogou_shuffle_train.json'
    valid_file = 'data/sogou_shuffle_valid.json'

    w, p, n = WebQACorpus.load_word_dictionary(baidu_file)
    word_dict, pos_dict, ner_dict = WebQACorpus.load_word_dictionary(train_file)
    torch.save([word_dict, pos_dict, ner_dict], open(args.dict_file, 'wb'))

    baidu_data = WebQACorpus(baidu_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
    train_data = WebQACorpus(train_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
    valid_data = WebQACorpus(valid_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)

    print("saving baidu_data ...")
    with open(baidu_file[:-4]+'pt', 'wb') as output:
        torch.save(baidu_data, output)

    print("saving train_data ...")
    with open(train_file[:-4]+'pt', 'wb') as output:
        torch.save(train_data, output)

    print("saving valid_data ...")
    with open(valid_file[:-4]+'pt', 'wb') as output:
        torch.save(valid_data, output)

def main():
    parser = argparse.ArgumentParser(description='Document Reader QA')
    add_argument(parser)
    args = parser.parse_args()
    preprocess_data(args)


if __name__ == "__main__":
    main()
