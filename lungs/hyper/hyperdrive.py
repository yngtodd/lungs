import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders
from lungs.models.lungXnet import LungXnet

import time
from lungs.utils.logger import print_progress
from lungs.meters import AverageMeter, mAPMeter


def train(epoch, train_loader, optimizer, criterion, model, args):
    """"""
    batch_time = AverageMeter(name='batch_time')
    loss_meter = AverageMeter(name='losses')
    mapmeter = mAPMeter()

    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        load_time.update(time.time() - end)

        bs, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w).cuda()
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), data.size(0))
        mapmeter.update(output, target)
      
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            print_progress('Train', epoch, args.num_epochs, batch_time, loss_meter, mapmeter)

    return loss_meter.avg


def validate(epoch, val_loader, criterion, model, args):
    """"""
    load_time = AverageMeter(name='loading_time')
    batch_time = AverageMeter(name='batch_time')
    loss_meter = AverageMeter(name='losses')
    mapmeter = mAPMeter()

    model.eval()
    end = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        load_time.update(time.time() - end)

        bs, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w).cuda()
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        output = model(data)
        output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, target)
 
        loss_meter.update(loss.item(), data.size(0))
        mapmeter.update(output, target)
      
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            print_progress('Validation', epoch, args.num_epochs, batch_time, loss_meter, mapmeter)
    
    return loss_meter.avg
    


def objective(hyperparams):
    """"""
    train_loss = AverageMeter(name='train_loss')
    val_loss = AverageMeter(name='val_losses')

    model = LungXnet()
    if args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()

    for epoch in range(1, args.num_epochs+1):
        trainloss = train(epoch, train_loader, optimizer, criterion, model, args)
        validloss = validate(epoch, val_loader, criterion, model, args)
        train_loss.update(trainloss)
        epoch_loss.update(validloss)

    train_loss.save(path=args.logspath)
    val_loss.save(path=args.logspath)
    return epoch_loss.avg
   
 
def main():
    global args
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Data loading
    loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
    global train_loader, val_loader
    train_loader = loaders.train_loader(imagetxt=args.traintxt)
    val_loader = loaders.val_loader(imagetxt=args.valtxt)
   
    global optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    global criterion 
    criterion = nn.BCELoss(size_average=True)
    if args.cuda:
        criterion.cuda()

    hparams = [(,)]


if __name__=="__main__":
    main()
