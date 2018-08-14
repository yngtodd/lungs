import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders

from lungs.models.miniencoder import Encoder
from lungs.models.miniencoder import Decoder
from lungs.models.miniencoder import AutoEncoder

import time
from lungs.utils.log import log_progress, record
from lungs.meters import AverageMeter, AUCMeter, mAPMeter


def save_checkpoint(state):
    filename='checkpoint' + str(state['epoch']) + '.pth.tar'
    torch.save(state, filename)


#@record
def train(train_loader, optimizer, criterion, model, meters, epoch, args):
    """"""
    loss_meter = meters['loss']
    batch_time = meters['train_time']
    num_samples = len(train_loader)

    model.train()
    print(f'epoch: {epoch}')
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        bs, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w)

        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
#        output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), data.size(0))
        end = time.time()

        if batch_idx % 10 == 0:
            save_checkpoint({
              'epoch': epoch,
              'state_dict': model.encoder.state_dict(),
              'optimizer' : optimizer.state_dict(),
            })

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            log_progress('Train', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter)


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

    encoder = Encoder() 
    decoder = Decoder()
    model = AutoEncoder(encoder, decoder)

    if args.cuda and not args.parallel:
        model.cuda()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()
    if args.cuda:
        criterion.cuda()

    meters = {
      'loss': AverageMeter(name='trainloss'),
      'train_time': AverageMeter(name='traintime'),
    }

    epoch_time = AverageMeter(name='epoch_time')
    end = time.time()
    print(f'Number of epochs: {args.num_epochs}')
    for epoch in range(1, args.num_epochs+1):
        train(train_loader, optimizer, criterion, model, meters, epoch, args)
        epoch_time.update(time.time() - end)
        end = time.time()

    print(f"\nJob's done! Total runtime: {epoch_time.sum}, Average runtime: {epoch_time.avg}")
    meters['loss'].save('/home/ygx/lungs/lungs/koda')


if __name__=="__main__":
    main()
