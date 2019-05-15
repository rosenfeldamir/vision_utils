
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from tensorboard_logger import log_value, Logger
import os
import os.path
import glob
from time import time
import shutil
from os.path import expanduser
homeDir = expanduser('~')

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, epoch, optimizer, maxIters=np.inf, train_loader=None,
          criterion=None, device='cuda:0'):
    T0 = time()
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    maxIters = min(maxIters, len(train_loader))
    startTime = time()
    nSamples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # target = target.long().squeeze()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # + criterion(output2,target2)
        # for testing accuracy on train...
        prec1 = accuracy(output.data, target.data, topk=(1,))[0]
        losses.update(loss.item())
        top1.update(prec1.item())
        nSamples += len(data)
        loss.backward()
        optimizer.step()
        nSamples += len(data)
        if time() - T0 > .5:  # update every half second
            T0 = time()
            elapsedTime = time() - startTime
            S = 'Train Epoch: \t{epoch} [{iters}/{total}] (\t{t2:.0f}%)]\tAvg Loss: \t{running_loss:.6f}\t({imgs_sec:.2f} imgs/sec)'
            S = S.format(epoch=epoch, iters=batch_idx * len(data),
                         total=len(train_loader.dataset), t2=100. * batch_idx / len(train_loader),
                         running_loss=losses.avg, imgs_sec=nSamples / elapsedTime)
            print('\r{}'.format(S), end="")
        if batch_idx > maxIters:
            print('stopping training early (debug mode')
            break
    return losses.avg, top1.avg


def test(model, epoch, test_loader=None,  criterion=None, maxIters=np.inf, device='cuda:0'):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    maxIters = min(maxIters, len(test_loader))
    for batch_idx, (data, target) in enumerate(tqdm(test_loader, total=len(test_loader))):
        data, target = data.to(device), target.to(device)
        output = model(data)
        prec1 = accuracy(output.data, target.data, topk=(1,))[0]
        loss = criterion(output, target)
        losses.update(loss.item())
        top1.update(prec1.item())
        if batch_idx >= maxIters:
            print('stopping validation early (debug mode')
            break
    P = '({epoch}) :[Test set] Avg. loss: \t{test_loss:.4f}, Acc: \t{cur_acc:.1f}%)'
    P = P.format(epoch=epoch, test_loss=losses.avg, cur_acc=top1.avg)
    print('\r{}'.format(P), end="")

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, epoch, modelDir):
    """Saves checkpoint to disk"""
    checkPointPath = '{}/{}'.format(modelDir, 'last.pth')
    torch.save(state, checkPointPath)
    if is_best:
        shutil.copyfile(checkPointPath, '{}/{}'.format(modelDir, 'best.pth'))


def get_num_params(model):
    return sum([np.prod(p.shape) for p in model.parameters()])


def trainAndTest(model, optimizer=None, modelDir=None, epochs=5, targetTranslator=None, model_save_freq=1,
                 train_loader=None, test_loader=None, stopIfPerfect=True, criterion=nn.CrossEntropyLoss(),
                 lr_scheduler=None, maxIters=np.inf, base_lr=.1, logger=None, device='cuda'):

    last_epoch = 0
    corrects = []

    needToSave = modelDir is not None and model_save_freq > 0
    all_accuracies = []
    all_train_accuracies = []
    all_val_losses = []
    all_train_losses = []
    if needToSave:
        if not os.path.isdir(modelDir):
            os.makedirs(modelDir)
        g = list(sorted(glob.glob(os.path.join(modelDir, '*'))))
        g = [g_ for g_ in g if not 'best' in g_]
        g_new = []
        for gg in g:  # fixing file names to be zero padded
            g1, g2 = os.path.split(gg)
            newName = '/'.join([g1, g2.zfill(4)])
            if gg != newName:
                print('moving')
                print(gg, 'to')
                print(newName)
                shutil.move(gg, newName)
            g_new.append(newName)
        g = list(sorted(g_new))
        if len(g) > 0:
            lastCheckpoint = g[-1]
            # load the last checkpoint
            print('loading from', lastCheckpoint)
            checkpoint = torch.load(lastCheckpoint)
            last_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            all_accuracies = checkpoint.get('all_accuracies', all_accuracies)
            all_train_accuracies = checkpoint.get(
                'all_train_accuracies', all_train_accuracies)
            all_train_losses = checkpoint.get(
                'all_train_losses', all_train_losses)
            all_val_losses = checkpoint.get('all_val_losses', all_val_losses)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(lastCheckpoint))
    best_acc = 0

    if lr_scheduler is not None:  # bring the lr scheduler up to date.
        lr_scheduler.last_epoch = last_epoch

    best_acc = 0

    for epoch in range(last_epoch, epochs):
        if lr_scheduler is not None:
            lr_scheduler.step()
        train_loss, train_acc = train(model=model, epoch=epoch, optimizer=optimizer,
                                      train_loader=train_loader, criterion=criterion,  maxIters=maxIters, device=device)
        all_train_losses.append(train_loss)
        all_train_accuracies.append(train_acc)
        test_loss, test_acc = test(model, epoch, test_loader=test_loader,
                                   criterion=criterion, maxIters=maxIters, device=device)
        all_val_losses.append(test_loss)
        all_accuracies.append(test_acc)

        is_best = False
        if test_acc > best_acc:
            best_acc = test_acc
            is_best = True

        if needToSave:
            if is_best or (epoch % model_save_freq) == 0 or (epoch == epochs - 1):
                print('\n\n******saving model******',)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'all_train_losses': all_train_losses,
                    'all_val_losses': all_val_losses,
                    'all_accuracies': all_accuracies,
                    'all_train_accuracies': all_train_accuracies,
                    'state_dict': model.state_dict(),
                    'num_params': get_num_params(model),
                }, is_best, epoch, modelDir)
    return {
        'epoch': epoch + 1,
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses,
        'all_accuracies': all_accuracies,
        'all_train_accuracies': all_train_accuracies,
        'state_dict': model.state_dict(),
        'num_params': get_num_params(model),
    }
