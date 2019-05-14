
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
modelsBaseDir = os.path.expanduser('~/models')
homeDir = expanduser('~')
experiment_base_dir = expanduser('~/experiments/')

def make_experiment_dir_name(experiment_params): 
    s = ''
    for i, (k, v) in enumerate(experiment_params.items()):
        s = s + k + '_' + str(v)
        if i < len(experiment_params):
            s += '_'
    return s

def make_experiment(experiment_name, experiment_params):
    exp_dir_name = make_experiment_dir_name(experiment_params)
    models_base = os.path.join(
        experiment_base_dir, experiment_name, 'checkpoints', exp_dir_name)
    os.makedirs(models_base, exist_ok=True)
    logger_path = os.path.join(
        experiment_base_dir, experiment_name, 'logs', exp_dir_name)
    os.makedirs(logger_path, exist_ok=True)
    return models_base, logger_path


def matVar(size=(1, 3, 64, 64), cuda=False):
    v = Variable(torch.randn(size))
    if cuda:
        v = v.cuda()
    return v

def train(model, epoch, optimizer, maxIters=np.inf, targetTranslator=None, train_loader=None, criterion=None, logger=None,
          device='cuda'):
    T0 = time()
    model.train()
    nBatches = 0
    running_loss = 0.0
    running_loss2 = 0.0
    losses = []
    nSamples = 0
    maxIters = min(maxIters, len(train_loader))
    startTime = time()
    test_accuracy = []
    correct = 0
    nSamples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if targetTranslator is not None:
            target2 = targetTranslator(target.clone())
            target2 = data.cuda(), target.cuda(), target2.cuda()
        target = target.long().squeeze()
        data = data.to(device)
        target = target.to(device)
        #, Variable(target2)
        optimizer.zero_grad()
        output = model(data)
        if type(output) is tuple:
            gates = output[1]
            output = output[0]

        loss = criterion(output, target)  # + criterion(output2,target2)
        # for testing accuracy on train...
        pred = output.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
        nSamples += len(data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        running_loss += loss.item()

        nBatches += 1  # len(data)
        nSamples += len(data)
        if batch_idx % 5 == 0 and time() - T0 > .2:
            T0 = time()
            elapsedTime = time() - startTime
                      
        
            S = 'Train Epoch: \t{epoch} [{iters}/{total} (\t{t2:.0f}%)]\tAvg Loss: \t{t3:.6f}\t({t5:.2f} imgs/sec)'.format(epoch=epoch, iters=batch_idx * len(data),
                                                                                                        total=len(train_loader.dataset),
                                                                                                        t2=100. * batch_idx / len(train_loader),
                                                                                                        t3 = running_loss / nBatches,
                                                                                                        t5 =nSamples / elapsedTime)

            if logger is not None:
                logger.log_value('training loss', loss.item(),
                                 batch_idx + epoch * maxIters)
            print('\r{}'.format(S), end="")
        if batch_idx > maxIters:
            break
    # b1
    if logger is not None:
        if hasattr(optimizer, 'param_groups'):

            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                logger.log_value('learning rate', cur_lr, epoch)

        if hasattr(optimizer, 'get_lr_factor'):
            logger.log_value('learning rate', optimizer.get_lr_factor(), epoch)
    return losses, correct / nSamples


def test(model, epoch, targetTranslator=None, test_loader=None, prev_acc=0, alpha=None, criterion=None, maxIters=np.inf, logger=None,
         device='cuda'):
    assert (criterion is not None)
    model.eval()
    test_loss = 0
    correct = 0
    nSamples = 0
    maxIters = min(maxIters, len(test_loader))
    for batch_idx, (data, target) in enumerate(tqdm(test_loader, total=len(test_loader))):
        target = target.long().squeeze()
        if targetTranslator is not None:
            target2 = targetTranslator(target.clone())
            target2 = target2.cuda()

        data, target = data.to(device), target.to(device)
        if alpha is not None:
            output = model(data, alpha)
        else:
            # b1
            output = model(data)
            if type(output) is tuple:
                gates = output[1]
                output = output[0]
        cur_test_loss = criterion(output, target).item()
        test_loss += cur_test_loss

        pred = output.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
        nSamples += len(data)
        if batch_idx >= maxIters:
            break

    # loss function already averages over batch size
    test_loss /= len(test_loader)
    if logger is not None:
        logger.log_value('test loss', test_loss, epoch)
    cur_acc = 100. * correct / nSamples
    # if prev_acc < cur_acc:
    P = '({}) :Test set: Avg. loss: \t{:.4f}, Acc: \t{}/{} ({:.1f}%)'.format(epoch,
                                                                             test_loss, correct, nSamples, cur_acc)
    if logger is not None:
        logger.log_value('test accuracy', cur_acc, epoch)

    print('\r{}'.format(P), end="")

    data.to('cpu')
    return 100. * correct / nSamples, test_loss


def save_checkpoint(state, is_best, epoch, modelDir):
    """Saves checkpoint to disk"""
    checkPointPath = '{}/{}'.format(modelDir, 'last.pth')
    torch.save(state, checkPointPath)
    if is_best:
        shutil.copyfile(checkPointPath, '{}/{}'.format(modelDir, 'best.pth'))


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

    for epoch in range(last_epoch, epochs):  # epochs + 1):

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses, train_acc = train(model=model, epoch=epoch, optimizer=optimizer, targetTranslator=targetTranslator,
                                  train_loader=train_loader, criterion=criterion,  maxIters=maxIters, device=device, logger=logger)

        cur_acc, cur_test_loss = test(model, epoch, targetTranslator=targetTranslator, test_loader=test_loader,
                                      prev_acc=best_acc, criterion=criterion, maxIters=maxIters, device=device, logger=logger)
        all_train_losses.append(losses[-1])
        all_val_losses.append(cur_test_loss)
        all_accuracies.append(cur_acc)
        all_train_accuracies.append(100 * train_acc)
        corrects.append(cur_acc)

        
        if needToSave and (epoch % model_save_freq == 0 or epoch == epochs - 1):
            print('saving model...',)
            checkPointPath = '{}/{}'.format(modelDir, epoch)
            if cur_acc > best_acc:
                best_acc = cur_acc
                is_best = True
            else:
                is_best = False
            save_checkpoint({
                'epoch': epoch + 1,
                'all_train_losses': all_train_losses,
                'all_val_losses': all_val_losses,
                'all_accuracies': all_accuracies,
                'all_train_accuracies': all_train_accuracies,
                'last_epoch_losses': losses,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'cur_acc': cur_acc,
                'train_acc': train_acc
            }, is_best, epoch, modelDir)
    # return the current state. 
    return {
                'epoch': epoch + 1,
                'all_train_losses': all_train_losses,
                'all_val_losses': all_val_losses,
                'all_accuracies': all_accuracies,
                'all_train_accuracies': all_train_accuracies,
                'last_epoch_losses': losses,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'cur_acc': cur_acc,
                'train_acc': train_acc
            }    
    
