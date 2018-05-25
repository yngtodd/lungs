import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders
from lungs.models.fp16_lungxnet import fp16_LungXnet

import time
from lungs.log import log_progress
from lungs.meters import AverageMeter, AUCMeter, mAPMeter

import logging
import logging.config


logging.config.fileConfig('logging.conf', defaults={'logfilename': './logs/main.log'})
logger = logging.getLogger(__name__)


def train(epoch, train_loader, optimizer, criterion, model, meters, args):
    """"""
    loss_meter = meters['train_loss']
    batch_time = meters['train_time']
    mapmeter = meters['train_mavep'] 
    num_samples = len(train_loader)

    model.train()
    end = time.time()
    print(f'Args.fp16 is {args.fp16}')
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.fp16:
            bs, c, h, w = data.size()
        else: #bs, n_crops, c, h, w = data.size()
            bs, c, h, w = data.size()
        data = data.view(-1, c, h, w)
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if args.fp16:
            data = data.half()
            target = target.half()
        optimizer.zero_grad()
        output = model(data)
        #print("output size", output.size(), "target size", target.size())
        #assert (output.data >= 0. & output.data <= 1.).all()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
  
        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), data.size(0))
        mapmeter.update(output, target)
        end = time.time()
      
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            log_progress('Train', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)
   

def validate(epoch, val_loader, criterion, model, meters, args):
    """"""
    loss_meter = meters['val_loss']
    batch_time = meters['val_time']
    mapmeter = meters['val_mavep']
    num_samples = len(val_loader)

    model.eval()
    end = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        if args.fp16:
            bs, c, h, w = data.size()
        else: bs, c, h, w = data.size()
        data = data.view(-1, c, h, w)
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if args.fp16:
            data = data.half()
            target = target.half()
        output = model(data)
        #print(f'output has size {output.size()}')
        #output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, target)

        batch_time.update(time.time() - end) 
        loss_meter.update(loss.item(), data.size(0))
        mapmeter.update(output, target)
        end = time.time()
      
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            log_progress('Validation', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)

    
def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if args.fp16:
        assert torch.backends.cudnn.enabled
    print("data loading started")
    	
    # Data loading
    loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
    train_loader = loaders.train_loader(imagetxt=args.traintxt)
    val_loader = loaders.val_loader(imagetxt=args.valtxt)
    print("data loaded ")
    model = fp16_LungXnet(num_layers=64, output_dim=14)
    if args.fp16:
        model=model.cuda().half()
        print("model loaded in half precision")
    elif args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
    else:
        model.cuda()


    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.BCEWithLogitsLoss(size_average=True)
    if args.cuda:
        criterion.cuda()
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
    for epoch in range(1, 3):
        train(epoch, train_loader, optimizer, criterion, model, train_meters, args)
        validate(epoch, val_loader, criterion, model, val_meters, args)
        epoch_time.update(time.time() - end)
        end = time.time()

    print(f"\nJob's done! Total runtime: {epoch_time.sum}, Average runtime: {epoch_time.avg}")


if __name__=="__main__":
    main()
