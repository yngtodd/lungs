import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

import datetime
from utils import utils
from utils.utils import AverageMeter, compute_auc
from collections import namedtuple
from collections import OrderedDict

# Starting Todd's imports
from lungs.models import LungXnet
from lungs.parser import parse_args


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_auc=0
_n_classes = 14

def main():
    global args, best_auc
    args = parse_args()
    #check if data folder exists
    args.tag = os.path.join('data/',args.tag)

    if args.trial:
        print('Running in trial mode with 15% of dataset..')
    if not os.path.exists(os.path.join(args.tag)):
        os.makedirs(os.path.join(args.tag))

    utils.setpath(args.tag)
    print('experiment tag {}'.format(args.tag))
    # model = LungXnet(num_classes=_n_classes)
    model = LungXnet(_n_classes)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    optimizers = {'SDG':torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay),
                 'Adam':torch.optim.Adam(model.parameters(),
                                lr=args.lr)}
    optimizer = optimizers[args.optimizer]

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, mode='min')
    criterion = nn.BCELoss(size_average=True).cuda()

    if args.resume:
        if os.path.isfile(os.path.join(args.tag,args.resume)):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.tag,args.resume))
            args.start_epoch = checkpoint['epoch']
            best_auc = checkpoint['best_auc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(args.tag,args.resume)))

    cudnn.benchmark = True

    dataloaders = utils.getloader(args)

    if args.evaluate:
        results=validate(dataloaders['test'], model, criterion)
        utils.save_pkl(f='test',obj=results)
        return
    epoch_time = AverageMeter(keep=True)
    epoch_train_loss = AverageMeter(keep=True)
    epoch_train_auc = AverageMeter(keep=True)
    epoch_val_loss = AverageMeter(keep=True)
    epoch_val_auc = AverageMeter(keep=True)

    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss,auc = train(dataloaders['train'], model, criterion, optimizer, epoch)
        epoch_train_loss.update(loss)
        epoch_train_auc.update(auc)
        # evaluate on validation set
        loss,auc= validate(dataloaders['val'], model, criterion, epoch)
        epoch_val_loss.update(loss)
        epoch_val_auc.update(auc)
        # remember best auc and save checkpoint
        is_best = epoch_val_auc.val > best_auc
        best_auc = max(epoch_val_auc.val, best_auc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_auc': best_auc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        epoch_time.update((time.time()-end)/60)
        end = time.time()
        scheduler.step(loss)
        print('epoch: [{0}/{1}]\t'
              'time {epoch_time.val:.3f} ({epoch_time.avg:.3f})\t'
              'train: loss {epoch_train_loss.val:.4f} ({epoch_train_loss.avg:.4f})\t'
              'auc {epoch_train_auc.val:.3f} ({epoch_train_auc.avg:.3f})]\t'
              'val: loss {epoch_val_loss.val:.4f} ({epoch_val_loss.avg:.4f})\t'
              'auc {epoch_val_auc.val:.3f} ({epoch_val_auc.avg:.3f})'.format(
               epoch, args.epochs,
               epoch_time=epoch_time,
               epoch_train_loss=epoch_train_loss,
               epoch_train_auc=epoch_train_auc,
               epoch_val_loss=epoch_val_loss,
               epoch_val_auc=epoch_val_auc))

        # #test
        # if epoch > 2:
        #     break

        utils.save_pkl(f='val',obj={'loss':epoch_val_loss.data,'auc':epoch_val_auc.data})
        utils.save_pkl(f='train',obj={'loss':epoch_train_loss.data,'auc':epoch_train_auc.data})


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    aucs = AverageMeter()

    # print('='*10+' Training '+'='*10+)

    # switch to train mode
    model.train()
    end = time.time()
    y_true, y_pred = torch.FloatTensor().cuda(), torch.FloatTensor().cuda()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input = input.cuda()
        # compute output

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        y_true = torch.cat([y_true,target.data],dim=0)
        y_pred = torch.cat([y_pred,output.data],dim=0)
        losses.update(loss.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and i > 0:
            # if i > 100:
            run_aucs = compute_auc(y_true=y_true, y_pred=y_pred,num_classes=_n_classes)
            aucs.update(run_aucs.mean(), y_true.size(0))
            print('train: [{0}][{1}/{2}]\t'
                  'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'aucs {aucs.val:.3f} ({aucs.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, aucs=aucs))
        #test
        if args.trial:
            if i > len(train_loader)//20:
                break
    train_auc = compute_auc(y_true=y_true, y_pred=y_pred, num_classes=_n_classes)
    aucs.update(train_auc.mean(),y_true.size(0))
    return losses.avg, aucs.val



def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    aucs = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        y_true, y_pred = torch.FloatTensor().cuda(), torch.FloatTensor().cuda()
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            bs, n_crops, c, h, w = input.size()
            input_view = input.view(-1, c, h, w).cuda()

            # compute output
            output = model(input_view)
            output_view = output.view(bs, n_crops, -1).mean(1)
            loss = criterion(output_view, target)

            # measure accuracy and record loss
            y_true = torch.cat([y_true, target.data], dim=0)
            y_pred = torch.cat([y_pred, output_view.data], dim=0)

            losses.update(loss.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:

                run_aucs = compute_auc(y_true=y_true, y_pred=y_pred, num_classes=_n_classes)
                aucs.update(run_aucs.mean(), y_true.size(0))

                print('val: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'aucs {aucs.val:.3f} ({aucs.avg:.3f})'.format(
                       epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                       aucs=aucs))

            #test
            if args.trial:
                if i > len(val_loader)//10:
                   break

    val_auc = compute_auc(y_true=y_true, y_pred=y_pred,num_classes=_n_classes)
    aucs.update(val_auc.mean(),y_true.size(0))
    if args.evaluate:
        return {'y_true':y_true.cpu().data.numpy(),
                'y_pred':y_pred.cpu().data.numpy(),
                'loss':losses.__dict__,
                'auc':val_auc}
    return losses.avg, val_auc.mean()


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def save_checkpoint(state, is_best):

    torch.save(state, os.path.join(args.tag,args.resume))
    if is_best:
        shutil.copyfile(os.path.join(args.tag,args.resume), os.path.join(args.tag,args.best_model))


if __name__ == '__main__':
    main()
