# -*- coding: utf-8 -*-

import torch
import codecs
from sklearn import metrics
from sklearn.utils import resample
import os
from sklearn.metrics import precision_score, accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import time
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    step_loss = []
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    
    n_classes = len(train_loader.dataset.classes)
    topk=5
    if n_classes <5:
        topk = n_classes
    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, targets) in tqdm(enumerate(train_loader), desc='train', leave=False):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu :
            images, targets = images.cuda(), targets.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, targets)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, topk))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        step_loss.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return losses, step_loss, top1.avg

    
@torch.no_grad()    
def evaluate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='')

    # switch to evaluate mode
    model.eval()
    n_classes = len(val_loader.dataset.classes)
    
    labels, scores = [], []
    begin_time = time.time()
    for i, (images, targets) in tqdm(enumerate(val_loader), desc='eval', leave=False):
        tmp_time = time.time()
        if args.gpu :
            images, targets = images.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, targets)
        # torch.cuda.synchronize()

        # measure accuracy and record loss
        score = torch.softmax(output, dim=1)
        acc1 = accuracy(output, targets, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        labels.append(targets)
        scores.append(score)

        # measure elapsed time
        batch_time.update(time.time() - tmp_time)
        

        if i % args.print_freq == 0:
            progress.print(i)
    end_time = time.time()   
     
    true_labels = torch.cat(labels, dim=0).cpu().numpy()
    scores = torch.cat(scores, dim=0).cpu().numpy()
    true_labels_onehot = label_binarize(true_labels, classes=np.arange(3))
        
    aucs = roc_auc_score(true_labels_onehot, scores, multi_class='ovr' , average=None)

    # TODO: this should also be done with the ProgressMeter
    print('Test time {}s , Acc@1 {top1.avg:.3f}'.format(int(end_time - begin_time) ,top1=top1))

    return top1.avg ,losses ,aucs


