#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/28
from __future__ import absolute_import


def add_data_argument(parser):
    parser.add_argument('-train-file', type=str, dest="train_file", default="data/train.all.json")
    parser.add_argument('-dev-file', type=str, dest="dev_file", default="valid_factoid_1.json")
    parser.add_argument('-test-file', type=str, dest="test_file", default=None)
    parser.add_argument('-save-file', type=str, dest="save_file", default=None)
    parser.add_argument('-load-file', type=str, dest="load_file", default=None)
    parser.add_argument('-topk', type=int, dest="topk", default=30000)


def add_argument(parser):
    # Train Option
    parser.add_argument('-epoch', type=int, dest="epoch", default=50)
    parser.add_argument('-batch', type=int, dest="batch", default=32)
    parser.add_argument('-device', type=int, dest="device", default=-1)
    parser.add_argument('-seed', type=int, dest="seed", default=1993)
    add_data_argument(parser)
    parser.add_argument('-exp-name', type=str, dest="exp_name", default=None, help="save model to model/$exp-name$/")
    parser.add_argument('-debug', dest="debug", action='store_true')

    # Model Option
    parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300)
    parser.add_argument('-pos-vec-size', type=int, dest="pos_vec_size", default=5)
    parser.add_argument('-ner-vec-size', type=int, dest="ner_vec_size", default=5)
    parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=128)
    parser.add_argument('-num-layers', type=int, dest='num_layers', default=2)
    parser.add_argument('-encoder-dropout', type=float, dest='encoder_dropout', default=0.3)
    parser.add_argument('-dropout', type=float, dest='dropout', default=0.3)
    parser.add_argument('-brnn', action='store_true', dest='brnn')
    parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='random')
    parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='LSTM', choices=["RNN", "GRU", "LSTM"])
    parser.add_argument('-multi-layer', type=str, dest='multi_layer_hidden', default='concatenate',
                        choices=["concatenate", "last"])

    # Optimizer Option
    parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
    parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adamax")
    parser.add_argument('-lr', type=float, dest="lr", default=0.02)
    parser.add_argument('-clip', type=float, default=9.0, dest="clip", help='clip grad by norm')
    parser.add_argument('-regular', type=float, default=0, dest="regular_weight", help='regular weight')


def get_folder_prefix(args, model):
    import os
    if args.exp_name is not None:
        model_folder = 'model' + os.sep + args.exp_name
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_prefix = model_folder + os.sep + args.exp_name
        with open(model_prefix + '.config', 'w') as output:
            output.write(model.__repr__())
            output.write(args.__repr__())
    else:
        model_folder = None
        model_prefix = None
    return model_folder, model_prefix


def get_data_dict(args):
    from .corpus.corpus import WebQACorpus
    import torch

    if args.load_file is None:
        word_dict, pos_dict, ner_dict = WebQACorpus.load_word_dictionary(args.train_file)
        word_dict.cut_by_top(args.topk)
        train_data = WebQACorpus(args.train_file, batch_size=50, device=-1,
                                 word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
        dev_data = WebQACorpus(args.dev_file, batch_size=50, device=-1,
                               word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
        if args.test_file is None:
            test_data = None
        else:
            test_data = WebQACorpus(args.test_file, batch_size=50, device=-1,
                                    word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
    else:
        word_dict, pos_dict, ner_dict, train_data, dev_data, test_data = torch.load(open(args.load_file, 'rb'))

    train_data.set_batch_size(args.batch)
    dev_data.set_batch_size(args.batch)
    train_data.set_device(args.device)
    dev_data.set_device(args.device)
    if test_data is not None:
        test_data.set_batch_size(args.batch)
        test_data.set_device(args.device)

    return word_dict, pos_dict, ner_dict, train_data, dev_data, test_data
