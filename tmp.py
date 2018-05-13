#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-05-13
import torch

pt_file = 'data/baidu_data-2.pt'
data = torch.load(open(pt_file, 'rb'))
data.set_batch_size(32)
data.set_device(-1)

batch_index = 0
for batch in data.next_batch():
    batch_index += 1
