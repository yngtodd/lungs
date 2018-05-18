import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders
from lungs.models.lungXnet import LungXnet

import time
from lungs.utils.logger import print_progress
from lungs.meters import AverageMeter, AUCMeter


def train(epoch, train_loader, optimizer, criterion, model, args):
    """"""
    load_time = AverageMeter(name='loading_time')
    batch_time = AverageMeter(name='batch_time')
    loss_meter = AverageMeter(name='losses')
    auc_meter = AverageMeter(name='aucs')

    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        load_time.update(time.time() - end)
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
 
        loss_meter.update(loss.item(), data.size(0))
        auc_meter.add(output, target)
        auc_meter.update()
      
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            print_progress('Train', epoch, args.num_epochs, batch_time, loss_meter, auc_meter)


def validate(epoch, val_loader, model, args):
    """"""
    load_time = AverageMeter(name='loading_time')
    batch_time = AverageMeter(name='batch_time')
    loss_meter = AverageMeter(name='losses')
    auc_meter = AverageMeter(name='aucs')

    model.eval()
    end = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        load_time.update(time.time() - end)
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
 
        loss_meter.update(loss.item(), data.size(0))
        auc_meter.add(output, target)
        auc_meter.update()
      
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            print_progress('Validation', epoch, args.num_epochs, batch_time, loss_meter, auc_meter)

    
def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Data loading
    loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
    train_loader = loaders.train_loader(imagetxt=args.traintxt, transform=False)
    val_loader = loaders.val_loader(imagetxt=args.valtxt, transform=False)

    model = LungXnet()
    if args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.BCELoss(size_average=True)
    if args.cuda:
        criterion.cuda()

    epoch_time = AverageMeter(name='epoch_time')
    end = time.time()

    for epoch in range(1, args.num_epochs+1):
        train(epoch, train_loader, optimizer, criterion, model, args)
        validate(epoch, val_loader, criterion, model, args)
        epoch_time.update(time.time() - end)
        end = time.time()

    print(f"\nJob's done! Total runtime: {epoch_time.sum}, Average runtime: {epoch_time.avg}")


if __name__=="__main__":
    main()
