# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/24
from __future__ import absolute_import
import time
import torch
import torch.nn as nn
from evaluate import evalutate
from model import DocumentReaderQA
import utils
from predict import predict_answer

args = utils.add_argument()

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
    torch.cuda.manual_seed(int(seed))

def get_data_dict(args, pt_file):
    data = torch.load(open(pt_file, 'rb'))
    data.set_batch_size(args.batch)
    data.set_device(args.device)
    return data

word_dict, pos_dict, ner_dict = torch.load(open(args.dict_file, 'rb'))
baidu_data = get_data_dict(args, args.baidu_data)
train_data = get_data_dict(args, args.train_data)
valid_data = get_data_dict(args, args.valid_data)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, loc: storage)
    print('load model from %s' % args.resume_snapshot)
else:
    model = DocumentReaderQA(word_dict, args, [pos_dict, ner_dict], [args.pos_vec_size, args.ner_vec_size])
    if args.word_vectors != 'random':
        model.embedding.load_pretrained_vectors(args.word_vectors, binary=True, normalize=args.word_normalize)
model.cnn_init(args)

model_folder, model_prefix = utils.get_folder_prefix(args, model)

if args.device >= 0:
    model.cuda(args.device)

params = list()
for name, param in model.named_parameters():
    print(name, param.size())
    params.append(param)

opt = getattr(torch.optim, args.optimizer)(params, lr=args.lr, weight_decay=args.regular_weight)


def eval_model(_model, _data):
    answer_dict, answer_dict_old, acc_s, acc_e, acc = predict_answer(_model, _data)
    q_level_p, char_level_f = evalutate(answer_dict)
    q_level_p_old, char_level_f_old = evalutate(answer_dict_old)
    return q_level_p, char_level_f, q_level_p_old, char_level_f_old, acc_s, acc_e, acc


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

        if batch_index % 500 == 0:
            print("iter: %d  %.2f  loss: %f" %(batch_index, batch_index/num_batch, loss.data[0]))

    print(forward_time, data_time, backward_time)
    return (loss_acc / num_batch)[0]


def eval_epoch(_model, _data):
    _model.eval()
    res = eval_model(model, valid_data)
    return res


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
    '''
    if iter_i % 3 == 0:
        train_loss = train_epoch(model, baidu_data)
    else:
        train_loss = train_epoch(model, train_data)
    '''

    train_loss = train_epoch(model, baidu_data)
    train_loss += train_epoch(model, train_data)
    train_end = time.time()

    model.eval()
    q_p, c_f, q_p_old, c_f_old, acc_s, acc_e, acc = eval_epoch(model, valid_data)
    eval_end = time.time()

    train_time = train_end - start
    eval_time = eval_end - train_end

    iter_str = "Iter %s" % iter_i
    time_str = "%s | %s" % (int(train_time), int(eval_time))
    train_loss_str = "Loss: %.2f" % train_loss
    acc_result = "Acc: %.2f Acc_s: %.2f Acc_e: %.2f" %(acc, acc_s, acc_e)
    eval_result = "Query Pre: %.2f: Char F1: %.2f" % (q_p, c_f)
    eval_result_old = "Query Pre: %.2f: Char F1: %.2f" % (q_p_old, c_f_old)
    log_str = ' | '.join([iter_str, time_str, train_loss_str, acc_result, eval_result, eval_result_old])

    print(log_str)
    if log_output is not None:
        log_output.write(log_str + '\n')
        log_output.flush()

    if model_prefix is not None:
        if best_loss > train_loss:
            torch.save(model, model_prefix + '.best.loss.model')
            best_loss = train_loss
        if best_cf < c_f_old:
            torch.save(model, model_prefix + '.best.char.f1.model')
            best_cf = c_f_old
        if best_qp < q_p_old:
            torch.save(model, model_prefix + '.best.query.pre.model')
            best_qp = q_p_old

if log_output is not None:
    log_output.write("Best Train Loss: %s\n" % best_loss)
    log_output.write("Best Char F1   : %s\n" % best_cf)
    log_output.write("Best QUery Pre : %s\n" % best_qp)
    log_output.close()
