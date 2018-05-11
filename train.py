# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/24
from __future__ import absolute_import
import time
from argparse import ArgumentParser
import torch
import torch.nn as nn
from evaluate import evalutate
from model import DocumentReaderQA
import utils
from predict import predict_answer

parser = ArgumentParser(description='Document Reader QA')
utils.add_argument(parser)
args = parser.parse_args()

if args.debug:
    args.train_file = "data/debug_data/baidu.debug.json"
    args.dev_file = "data/debug_data/sogou.debug.json"

if args.seed < 0:
    seed = time.time() % 10000
else:
    seed = args.seed
print("Random Seed: %d" % seed)
torch.manual_seed(int(seed))

if args.device >= 0:
    torch.cuda.set_device(args.device)

word_dict, pos_dict, ner_dict, train_data, dev_data, test_data = utils.get_data_dict(args)

model = DocumentReaderQA(word_dict, args, [pos_dict, ner_dict], [args.pos_vec_size, args.ner_vec_size])

model_folder, model_prefix = utils.get_folder_prefix(args, model)

if args.device >= 0:
    model.cuda(args.device)

if args.word_vectors != 'random':
    model.embedding.load_pretrained_vectors(args.word_vectors, binary=True, normalize=args.word_normalize)

params = list()
for name, param in model.named_parameters():
    print(name, param.size())
    params.append(param)

opt = getattr(torch.optim, args.optimizer)(params, lr=args.lr, weight_decay=args.regular_weight)


def eval_model(_model, _data):
    answer_dict = predict_answer(_model, _data)
    q_level_p, char_level_f = evalutate(answer_dict)
    return q_level_p, char_level_f


def train_epoch(_model, _data):
    model.train()
    loss_acc = 0
    num_batch = len(_data) / args.batch
    batch_index = 0
    forward_time = 0
    data_time = 0
    backward_time = 0
    back_time = time.time()
    for batch in _data.next_batch():
        batch_index += 1
        data_time = time.time() - back_time
        opt.zero_grad()

        start_time = time.time()
        loss = model.loss(batch)
        end_time = time.time()
        forward_time += end_time - start_time
        loss.backward()
        loss_acc += loss.data

        if args.clip > 0:
            nn.utils.clip_grad_norm(model.parameters(), args.clip)

        opt.step()
        back_time = time.time()
        backward_time += back_time - end_time

        if batch_index % 100 == 0:
            print("iter: %d  %.2f  loss: %f" %(batch_index, batch_index/num_batch, loss.data[0]))

    print(forward_time, data_time, backward_time)
    return (loss_acc / num_batch)[0]


def eval_epoch(_model, _data):
    _model.eval()
    q_p, c_f = eval_model(model, dev_data)
    return q_p, c_f


print("training")
best_loss = 200.
best_cf = 0.
best_qp = 0.

if model_prefix is not None:
    log_output = open(model_prefix + '.log', 'w')
else:
    log_output = None

for iter_i in range(args.epoch):
    start = time.time()

    model.train()
    train_loss = train_epoch(model, train_data)
    train_end = time.time()

    model.eval()
    q_p, c_f = eval_epoch(model, dev_data)
    eval_end = time.time()

    train_time = train_end - start
    eval_time = eval_end - train_end

    iter_str = "Iter %s" % iter_i
    time_str = "%s | %s" % (int(train_time), int(eval_time))
    train_loss_str = "Loss: %.2f" % train_loss
    eval_result = "Query Pre: %.2f: Char F1: %.2f" % (q_p, c_f)
    log_str = ' | '.join([iter_str, time_str, train_loss_str, eval_result])

    print(log_str)
    if log_output is not None:
        log_output.write(log_str + '\n')
        log_output.flush()

    if model_prefix is not None:
        if best_loss > train_loss:
            torch.save([word_dict, pos_dict, ner_dict], model_prefix + '.best.loss.dict')
            torch.save(model, model_prefix + '.best.loss.model')
            best_loss = train_loss
        if best_cf < c_f:
            torch.save([word_dict, pos_dict, ner_dict], model_prefix + '.best.char.f1.dict')
            torch.save(model, model_prefix + '.best.char.f1.model')
            best_cf = c_f
        if best_qp < q_p:
            torch.save([word_dict, pos_dict, ner_dict], model_prefix + '.best.query.pre.dict')
            torch.save(model, model_prefix + '.best.query.pre.model')
            best_qp = q_p

if log_output is not None:
    log_output.write("Best Train Loss: %s\n" % best_loss)
    log_output.write("Best Char F1   : %s\n" % best_cf)
    log_output.write("Best QUery Pre : %s\n" % best_qp)
    log_output.close()