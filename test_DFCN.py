#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from tensorboardX import SummaryWriter
import dataset
from DFCN import DFCN

class Test(object):
    def __init__(self, Dataset, Network, path, ss):
        self.cfg = Dataset.Config(datapath=path, snapshot=ss, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)

        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
    
    def save(self):
        time_t = 0.0
        numbeR = 0
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                time_start = time.time()

                up_edge, up_sal, up_sal_f = self.net(image, shape)

                out = up_sal_f[-1]
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                head  = './result/'+ self.cfg.datapath.split('/')[-1]

                numbeR += 1
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))

                time_end = time.time()

                print(head, time_end - time_start, numbeR)
                time_t = time_t + time_end - time_start

        print("The total time is %s seconds!" % (time_t))

if __name__=='__main__':
    for path in ['./data/test/ECSSD', './data/test/PASCAL-S', './data/test/DUTS-test', './data/test/HKU-IS', './data/test/DUT-OMRON']:
        t = Test(dataset, DFCN, path, './model/DFCN-80')
        t.save()
