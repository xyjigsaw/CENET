# Name: test
# Author: Reacubeth
# Time: 2021/8/25 10:51
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*
import argparse
import numpy as np
import torch
import pickle
import time
import datetime
import os
import random
import utils
from cenet_model import CENET


def execute_test(args, total_data, model,
                 data,
                 s_history, o_history,
                 s_label, o_label,
                 s_frequency, o_frequency):
    s_ranks1 = []
    o_ranks1 = []
    all_ranks1 = []

    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    total_data = utils.to_device(torch.from_numpy(total_data))
    for batch_data in utils.make_batch(data,
                                       s_history,
                                       o_history,
                                       s_label,
                                       o_label,
                                       s_frequency,
                                       o_frequency,
                                       args.batch_size):
        batch_data[0] = utils.to_device(torch.from_numpy(batch_data[0]))
        batch_data[3] = utils.to_device(torch.from_numpy(batch_data[3])).float()
        batch_data[4] = utils.to_device(torch.from_numpy(batch_data[4])).float()
        batch_data[5] = utils.to_device(torch.from_numpy(batch_data[5])).float()
        batch_data[6] = utils.to_device(torch.from_numpy(batch_data[6])).float()

        with torch.no_grad():
            sub_rank1, obj_rank1, cur_loss1, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Test', total_data)

            s_ranks1 += sub_rank1
            o_ranks1 += obj_rank1
            tmp1 = sub_rank1 + obj_rank1
            all_ranks1 += tmp1

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
    return s_ranks1, o_ranks1, all_ranks1, \
           s_ranks2, o_ranks2, all_ranks2, \
           s_ranks3, o_ranks3, all_ranks3
