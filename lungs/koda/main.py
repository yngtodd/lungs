import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders

from lungs.koda import AutoEncoder
from lungs.koda import DenseNet121
from lungs.koda import LinearDecoder

import time
from lungs.utils.log import log_progress, record
from lungs.meters import AverageMeter, AUCMeter, mAPMeter


@record
def train(train_loader, optimizer, criterion, model, meters, args, epoch):
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
            log_progress('Train', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)


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

    encoder = DenseNet121()
    decoder = LinearDecoder(200, 300)
    model = AutoEncoder(encoder, decoder)

    if args.cuda and not args.parallel:
        model.cuda()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()
    if args.cuda:
        criterion.cuda()

    train_meters = {
      'train_loss': AverageMeter(name='trainloss'),
      'train_time': AverageMeter(name='traintime'),
      'train_mavep': mAPMeter()
    }

    epoch_time = AverageMeter(name='epoch_time')
    end = time.time()
    print(f'Number of epochs: {args.num_epochs}')
    for epoch in range(1, args.num_epochs+1):
        train(train_loader, optimizer, criterion, model, train_meters, args, epoch=epoch)
        epoch_time.update(time.time() - end)
        end = time.time()

    print(f"\nJob's done! Total runtime: {epoch_time.sum}, Average runtime: {epoch_time.avg}")


if __name__=="__main__":
    main()
