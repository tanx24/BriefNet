#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import dataset
from DFCN import DFCN
from apex import amp
import loss
import os

bce2d_new = loss.bce2d_new
src_loss = loss.SRCLoss(reduction='elementwise_mean')
dice_loss = loss.Diceloss()

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='./data/train/', savepath='./model', mode='train', batch=32, lr=0.005, momen=0.9, decay=5e-4, epoch=80)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net = net.cuda()
    ## see the train set
    train_path = 'train_res'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')

    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, edge) in enumerate(loader):
            image, mask, edge = image.cuda().float(), mask.cuda().float(), edge.cuda().float()

            up_edge, up_sal, up_sal_f = net(image)
            # edge part
            edge_loss = []

            for ii, ix in enumerate(up_edge):
                edge_loss.append(bce2d_new(ix, edge, reduction='elementwise_mean'))
            edge_loss = sum(edge_loss)

            # sal part
            sal_loss1 = []
            sal_loss2 = []

            for ii, ix in enumerate(up_sal):
                sal_loss1.append(F.binary_cross_entropy_with_logits(ix, mask, reduction='elementwise_mean') + src_loss(ix, mask, edge) + dice_loss(ix, mask))
            
            for ii, ix in enumerate(up_sal_f):
                sal_loss2.append(F.binary_cross_entropy_with_logits(ix, mask, reduction='elementwise_mean') + src_loss(ix, mask, edge) + dice_loss(ix, mask))
           
            sal_loss = sum(sal_loss1) + sum(sal_loss2)
            
            loss = sal_loss + edge_loss

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                vutils.save_image(torch.sigmoid(up_sal_f[-1].data), train_path + '/epoch%04d-iter%d-sal-c.jpg' % (epoch, step),normalize=True, padding=0)
                vutils.save_image(image.data, train_path + '/epoch%04d-iter%d-sal-data.jpg' % (epoch, step), padding=0)
                vutils.save_image(mask.data, train_path + '/epoch%04d-iter%d-sal-target.jpg' % (epoch, step), padding=0)

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss':loss.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if epoch>cfg.epoch/4*3:
            torch.save(net.state_dict(), cfg.savepath+'/DFCN-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, DFCN)