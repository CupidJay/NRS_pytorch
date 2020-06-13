import os
import numpy as np
import shutil
import logging
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from DataLoader import get_dataset
from sklearn.svm import LinearSVC, SVC
from sklearn .preprocessing import normalize
from utils.loading import *

def test(model, cfg, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if cfg.MODEL.WEIGHT:
        if os.path.isfile((cfg.MODEL.WEIGHT)):
            logger.info("=> loading checkpoint '{}'".format(cfg.MODEL.RESUME))
            checkpoint = torch.load(cfg.MODEL.RESUME)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            logger.info("=> loaded checkpoint '{}' (epoch {}, acc {})".format(cfg.MODEL.RESUME, start_epoch, best_acc))
    elif cfg.MODEL.RESUME:
        if os.path.isfile((cfg.MODEL.RESUME)):
            logger.info("=> loading checkpoint '{}'".format(cfg.MODEL.RESUME))
            checkpoint = torch.load(cfg.MODEL.RESUME)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            logger.info("=> loaded checkpoint '{}' (epoch {}, acc {})".format(cfg.MODEL.RESUME, start_epoch, best_acc))
    else:
        logger.info("=> no checkpoint found at '{}'".format(cfg.MODEL.RESUME))

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataset(cfg)

    #logger.info("finish build model\n{}".format(model))

    loss_func = torch.nn.CrossEntropyLoss()

    logger.info("start testing")

    # test model accuracy
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    eval_total = 0.
    time_list = []
    for i, (batch_x, batch_y) in enumerate(val_loader):
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        if len(batch_x.size()) > 4:
            bs, crops, ch, h, w = batch_x.size()
            torch.cuda.synchronize()
            start = time.time()
            out= model(batch_x.view(-1, ch, h, w))
            torch.cuda.synchronize()
            end = time.time()
            out = out.view(bs, crops, -1).mean(dim=1)
        else:
            #torch.cuda.synchronize()
            start = time.time()
            out = model(batch_x)
            #torch.cuda.synchronize()
            end = time.time()
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum().item()
        eval_acc += num_correct
        eval_total += batch_y.size(0)
        time_list.append(end-start)
        if (i+1)%100==0:
            print(i)
        if i>1000:
            break
    eval_loss = eval_loss / eval_total
    eval_acc = 100 * float(eval_acc) / float(eval_total)

    logger.info("Test Loss: {:.6f}, Acc: {:.6f}%".format(eval_loss, eval_acc))
    print(time_list)
    logger.info("Time: {}".format(np.sum(time_list)))
    return np.sum(time_list)
