#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/28
from __future__ import absolute_import
from argparse import ArgumentParser


def add_argument():
    parser = ArgumentParser(description='Document Reader QA')
    # Data Option
    parser.add_argument('-baidu-file', type=str, dest="baidu_file", default="data/baidu_data.json")
    parser.add_argument('-baidu-data', type=str, dest="baidu_data", default="data/baidu_data.pt")
    parser.add_argument('-train-file', type=str, dest="train_file", default="data/sogou_shuffle_train.json")
    parser.add_argument('-train-data', type=str, dest="train_data",
                        default="data/sogou_shuffle_train.pt")
    parser.add_argument('-valid-file', type=str, dest="valid_file", default="data/sogou_shuffle_valid.json")
    parser.add_argument('-valid-data', type=str, dest="valid_data",
                        default="data/sogou_shuffle_valid.pt")
    parser.add_argument('-test-file', type=str, dest="test_file", default=None)
    parser.add_argument('-topk', type=int, dest="topk", default=30000)
    parser.add_argument('-dict', type=str, dest="dict_file", default='data/vocab.pt')

    # Train Option
    parser.add_argument('-epoch', type=int, dest="epoch", default=50)
    parser.add_argument('-batch', type=int, dest="batch", default=32)
    parser.add_argument('-device', type=int, dest="device", default=-1)
    parser.add_argument('-seed', type=int, dest="seed", default=1993)
    parser.add_argument('-exp-name', type=str, dest="exp_name", default=None, help="save model to model/$exp-name$/")
    parser.add_argument('-debug', dest="debug", action='store_true')
    parser.add_argument('-resume_snapshot', type=str, dest='resume_snapshot', default=None)

    # Model Option
    parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300)
    parser.add_argument('-pos-vec-size', type=int, dest="pos_vec_size", default=5)
    parser.add_argument('-ner-vec-size', type=int, dest="ner_vec_size", default=5)
    parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=128)
    parser.add_argument('-num-layers', type=int, dest='num_layers', default=2)
    parser.add_argument('-encoder-dropout', type=float, dest='encoder_dropout', default=0.3)
    parser.add_argument('-dropout', type=float, dest='dropout', default=0.3)
    parser.add_argument('-loss_margin', type=float, dest='loss_margin', default=1)
    parser.add_argument('-brnn', action='store_true', dest='brnn')
    parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='random')
    parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='LSTM', choices=["RNN", "GRU", "LSTM"])
    parser.add_argument('-multi-layer', type=str, dest='multi_layer_hidden', default='concatenate',
                        choices=["concatenate", "last"])
    parser.add_argument('-channels', type=int, dest='channels', default=1)

    # Optimizer Option
    parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
    parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adamax")
    parser.add_argument('-lr', type=float, dest="lr", default=0.02)
    parser.add_argument('-clip', type=float, default=9.0, dest="clip", help='clip grad by norm')
    parser.add_argument('-regular', type=float, default=0, dest="regular_weight", help='regular weight')

    # Predict Option
    parser.add_argument('-model', type=str, dest="model_file", default=None)
    parser.add_argument('-test', type=str, dest="test_file", default='data/sogou_shuffle_valid.json')
    parser.add_argument('-output', type=str, dest="out_file", default='output/result')
    parser.add_argument('-question', action='store_true', dest="question")

    args = parser.parse_args()
    return args

def get_folder_prefix(args, model):
    import os
    if args.exp_name is not None:
        model_folder = 'saved_checkpoint' + os.sep + args.exp_name
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
