import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders
from lungs.models.lungXnet import LungXnet

import time
from lungs.utils.log import log_progress, record
from lungs.meters import AverageMeter, AUCMeter, mAPMeter

from lungs.log import log_progress, record
from lungs.meters import AverageMeter, AUCMeter, mAPMeter

import logging
import logging.config


@record
def train(epoch, train_loader, optimizer, criterion, model, meters, args):
    """"""
    loss_meter = meters['train_loss']
    batch_time = meters['train_time']
    mapmeter = meters['train_mavep']
    num_samples = len(train_loader)

    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        bs, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w)

        if args.cuda:
            data = data.cuda(non_blocking=True).half()
            target = target.cuda(non_blocking=True).half()

        optimizer.zero_grad()
        output = model(data)
        output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), data.size(0))
        mapmeter.update(output, target)
        end = time.time()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            log_progress('Train', epoch, args.num_epochs, batch_time, loss_meter, mapmeter)

    return loss_meter.avg, mapmeter.avg
   

@record
def validate(epoch, val_loader, criterion, model, meters, args):
    """"""
    loss_meter = meters['val_loss']
    batch_time = meters['val_time']
    mapmeter = meters['val_mavep']
    num_samples = len(val_loader)

    model.eval()
    end = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        bs, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w)

        if args.cuda:
            data = data.cuda(non_blocking=True).half()
            target = target.cuda(non_blocking=True).half()

        output = model(data)
        output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, target)

        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), data.size(0))
        mapmeter.update(output, target)
        end = time.time()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            log_progress('Validation', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)

    return loss_meter.avg, mapmeter.avg


def main():
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Data loading
    loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
    train_loader = loaders.train_loader(imagetxt=args.traintxt)
    val_loader = loaders.val_loader(imagetxt=args.valtxt)

    model = LungXnet()
    if args.parallel:
        model = nn.DataParallel(model)
        model = model.cuda().half()

    if args.cuda and not args.parallel:
        model.cuda().half()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    criterion = nn.BCEWithLogitsLoss(size_average=True)
    if args.cuda:
        criterion.cuda()

    train_meters = {
      'train_loss': AverageMeter(name='trainloss'),
      'train_time': AverageMeter(name='traintime'),
      'train_mavep': mAPMeter()
    }

    val_meters = {
      'val_loss': AverageMeter(name='valloss'),
      'val_time': AverageMeter(name='valtime'),
      'val_mavep': mAPMeter()
    }

    logger.info(f'Starting off!')
    epoch_time = AverageMeter(name='epoch_time')
    end = time.time()
    print(f'Number of epochs: {args.num_epochs}')
    for epoch in range(1, args.num_epochs+1):
        train_loss, train_map = train(epoch, train_loader, optimizer, criterion, model, train_meters, args)
        val_loss, val_map = validate(epoch, val_loader, criterion, model, val_meters, args)
        epoch_time.update(time.time() - end)
        end = time.time()

    print(f"\nJob's done! Total runtime: {epoch_time.sum}, Average runtime: {epoch_time.avg}")


if __name__=="__main__":
    main()
