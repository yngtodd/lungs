import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from parser import parse_args
from data.loaders import XRayLoaders
from models.hj_fp16 import hj_fp16

import time
#from lungs.log import log_progress
#from lungs.meters import AverageMeter, AUCMeter, mAPMeter

#import logging
#import logging.config


#logging.config.fileConfig('logging.conf', defaults={'logfilename': './logs/main.log'})
#logger = logging.getLogger(__name__)


def train(epoch, train_loader, optimizer, criterion, model, args):
    """"""
    #loss_meter = meters['train_loss']
    #batch_time = meters['train_time']
    #mapmeter = meters['train_mavep'] 
    num_samples = len(train_loader)

    model.train()
    end = time.time()
    print(f'Args.fp16 is {args.fp16}')
    for batch_idx, (data, target) in enumerate(train_loader):
        print("data shape",data.size())
        if args.fp16:
            bs, c, h, w = data.size()
            data= data.view(-1,c,h,w)
        else: #bs, n_crops, c, h, w = data.size()
            bs,c, h, w = data.size()
        data = data.view(-1, c, h, w)
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if args.fp16:
            data = data.half()
            target = target.half()
        optimizer.zero_grad()
        output = model(data)
        #output = output.view(bs,n_crops, -1).mean(1)
        #print("output size", output.size(), "target size", target.size())
        #assert (output.data >= 0. & output.data <= 1.).all()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
  
        #batch_time.update(time.time() - end)
        #loss_meter.update(loss.item(), data.size(0))
        #mapmeter.update(output, target)
        #end = time.time()
      
        #if batch_idx % args.log_interval == 0 and batch_idx > 0:
            #log_progress('Train', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)
    print("time taken for training epoch",time.time()-end)   

def validate(epoch, val_loader, criterion, model, args):
    """"""
    #loss_meter = meters['val_loss']
    #batch_time = meters['val_time']
    #mapmeter = meters['val_mavep']
    num_samples = len(val_loader)

    model.eval()
    end = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        if args.fp16:
            bs,n_crops, c, h, w = data.size()
            data = data.view(-1,c,h,w)
        else: bs, c, h, w = data.size()
        data = data.view(-1, c, h, w)
        
        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if args.fp16:
            data = data.half()
            target = target.half()
        output = model(data)
        print(f'output has size {output.size()}')
        output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, target)

        #batch_time.update(time.time() - end) 
        #loss_meter.update(loss.item(), data.size(0))
        #mapmeter.update(output, target)
        #end = time.time()
      
        #if batch_idx % args.log_interval == 0 and batch_idx > 0:
            #log_progress('Validation', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)
    print("time taken for validation epoch",time.time()-end)
    
def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if args.fp16:
        assert torch.backends.cudnn.enabled
    print("data loading started")
    	
    # Data loading
    loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
    train_loader = loaders.train_loader(imagetxt=args.traintxt)
    #val_loader = loaders.val_loader(imagetxt=args.valtxt)
    print("data loaded ")
    model = hj_fp16(num_layers=64, output_dim=14)
    if args.fp16 and args.parallel:
        model = nn.DataParallel(model)
        model=model.cuda().half()
        print("model loaded in half precision and running in parallel")
    elif args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        print("model loaded in single precision and in data parallel mode")
    elif args.fp16:
        model=model.cuda().half()
        print("model loaded in half precision")
    else:
        model.cuda()

    print(model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    criterion = nn.BCEWithLogitsLoss()
    if args.cuda:
        criterion.cuda()
    criterion.cuda()
    '''
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
    '''
    start = time.time()
    for epoch in range(1, args.num_epochs):
        train(epoch, train_loader, optimizer, criterion, model, args)
        #validate(epoch, val_loader, criterion, model, args)
    
    print("\nJob's done! Total runtime:",time.time()-start)


if __name__=="__main__":
    main()
