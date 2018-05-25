#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-05-21
import torch
data = torch.load('../drqa-classify/data/sogou_shuffle_train.pt')
data.set_batch_size(25)
index = 0
for batch in data.next_batch():
    if index > 4:break
    print(batch.q_keys)
    print(batch.e_keys)
    print(batch.start_position)
