import os
import shutil
import logging
import time
import torch
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from DataLoader import get_dataset

def train(model, cfg, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = cfg.MODEL.META_ARCHITECTURE

    #used for freeze parameters
    if cfg.SOLVER.FREEZE:
        if model_type.startswith('VGG'):
            for param in model.features.parameters():
                param.requires_grad = False
        elif model_type.startswith('ResNet'):
            for i, m in enumerate(model.children()):
                if i>8:
                    break
                for param in m.parameters():
                    param.requires_grad = False
        logger.info("freeze conv1-5 parameters")
    else:
        logger.info("all parameters updated")

    if cfg.SOLVER.OPTIMIZER=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR,
                                   momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.LR_SCHEDULER_MILESTONE,
                            gamma=cfg.SOLVER.LR_SCHEDULER_GAMMA)
    #scheduler = StepLR(optimizer, step_size=30, gamma=cfg.SOLVER.LR_SCHEDULER_GAMMA)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    start_epoch = 0
    best_acc = 0.

    if cfg.MODEL.WEIGHT:
        if os.path.isfile((cfg.MODEL.WEIGHT)):
            checkpoint = torch.load(cfg.MODEL.WEIGHT)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(cfg.MODEL.WEIGHT))
    elif cfg.MODEL.RESUME:
        if os.path.isfile((cfg.MODEL.RESUME)):
            logger.info("=> loading checkpoint '{}'".format(cfg.MODEL.RESUME))
            checkpoint = torch.load(cfg.MODEL.RESUME)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            #scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.MODEL.RESUME, start_epoch))
    else:
        logger.info("=> no checkpoint found at '{}'".format(cfg.MODEL.RESUME))

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataset(cfg)
    logger.info("finish build model\n{}".format(model))

    loss_func = torch.nn.CrossEntropyLoss()
    loss_func.to(device)

    logger.info("start training for {} epochs".format(cfg.SOLVER.MAX_EPOCHS))

    is_best = False
    best_epoch = start_epoch
    eval_acc = 0.

    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []

    forward_time = 0.
    backward_time = 0.

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_loss = 0.
        train_acc = 0.
        train_total = 0.

        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = model(batch_x)
            torch.cuda.synchronize()
            end = time.time()

            forward_time += end-start

            if isinstance(out, list):
                loss = 0.
                for pred_y in out:
                    loss = loss+loss_func(pred_y, batch_y)
                out = out[-1]
            else:
                loss = loss_func(out, batch_y)

            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum().item()
            train_acc += train_correct
            train_total += batch_y.size(0)
            optimizer.zero_grad()
            torch.cuda.synchronize()
            start = time.time()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            backward_time += end-start

        train_loss = train_loss / train_total
        train_acc = 100 * float(train_acc) / float(train_total)

        if cfg.SOLVER.LR_SCHEDULER_ON:
            scheduler.step()

        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        eval_total = 0.
        for batch_x, batch_y in val_loader:
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if len(batch_x.size()) > 4:
                bs, crops, ch, h, w = batch_x.size()
                out = model(batch_x.view(-1, ch, h, w))
                out = out.view(bs, crops, -1).mean(dim=1)
            else:
                out = model(batch_x)

            if isinstance(out, list):
                loss = 0.
                for pred_y in out:
                    loss = loss+loss_func(pred_y, batch_y)
                out = out[-1]
            else:
                loss = loss_func(out, batch_y)

            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum().item()
            eval_acc += num_correct
            eval_total += batch_y.size(0)
        eval_loss = eval_loss / eval_total
        eval_acc = 100 * float(eval_acc) / float(eval_total)

        is_best = False
        if eval_acc>best_acc:
            is_best = True
            best_acc = eval_acc
            best_epoch = epoch+1

        filename = []
        filename.append(os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar'))
        filename.append(os.path.join(cfg.OUTPUT_DIR, 'model_best.pth.tar'))

        save_checkpoint({
            'epoch': epoch+1,
            'arch': cfg.MODEL.META_ARCHITECTURE,
            'state_dict': model.state_dict(),
            'acc': eval_acc,
            #'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, filename)

        if (epoch + 1) % cfg.LOG_EPOCHS == 0:
            logger.info("epoch {}".format(epoch + 1))
            logger.info("Train Loss: {:.6f}, Acc: {:.6f}%".format(train_loss, train_acc))
            logger.info("Test Loss: {:.6f}, Acc: {:.6f}%".format(eval_loss, eval_acc))
            logger.info('save model into {}'.format(filename[0]))
            train_losses.append(train_loss)
            eval_losses.append(eval_loss)
            train_accs.append(train_acc)
            eval_accs.append(eval_acc)

    torch.cuda.synchronize()
    end = time.time()
    logger.info('forward time is {}, backward time is {}'.format(forward_time, backward_time))

    logger.info('Best at epoch %d, test_accuracy %f' % (best_epoch, best_acc))

    return model, eval_acc, best_acc


def eval(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = torch.nn.CrossEntropyLoss()
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    eval_total = 0.
    for batch_x, batch_y in data_loader:
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if len(batch_x.size()) > 4:
            bs, crops, ch, h, w = batch_x.size()
            out= model(batch_x.view(-1, ch, h, w))
            out = out.view(bs, crops, -1).mean(dim=1)
        else:
            out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum().item()
        eval_acc += num_correct
        eval_total += batch_y.size(0)
    eval_loss = eval_loss / eval_total
    eval_acc = 100 * float(eval_acc) / float(eval_total)
    return eval_loss, eval_acc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, cfg):
    lr = cfg.SOLVER.LR * (0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr