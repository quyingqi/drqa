#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/28
from __future__ import absolute_import
import torch
import argparse
from utils import get_data_dict, add_data_argument


def preprocess_data(args):
    args.load_file = None
    args.batch = 64
    args.device = -1
    word_dict, pos_dict, ner_dict, train_data, dev_data, test_data = get_data_dict(args)
    print("save data to %s ..." % args.save_file)
    with open(args.save_file, 'wb') as output:
        torch.save([word_dict, pos_dict, ner_dict, train_data, dev_data, test_data], output)


def main():
    parser = argparse.ArgumentParser(description='Document Reader QA')
    add_data_argument(parser)
    args = parser.parse_args()
    preprocess_data(args)


if __name__ == "__main__":
    main()
