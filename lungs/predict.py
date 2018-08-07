import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders
from lungs.koda import DenseNet121
from lungs.models.transfer import TransferNet

import os
import time
from lungs.utils.log import log_progress, record
from lungs.meters import AverageMeter, AUCMeter, mAPMeter


def save_checkpoint(state):
    savedir = '/home/ygx/lungs/lungs/transfersaves'
    filename= 'checkpoint' + str(state['epoch']) + '.pth.tar'
    path = os.path.join(savedir, filename)
    torch.save(state, path)


#@record
def validate(val_loader, criterion, model, meters, args, epoch):
    """"""
    loss_meter = meters['val_loss']
    batch_time = meters['val_time']
    mapmeter = meters['val_mavep']
    accumeter = meters['val_accuracy']
    num_samples = len(val_loader)

    model.eval()
    correct = 0
    end = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        bs, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w)

        if args.cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        output = model(data)
        output = output.view(bs, n_crops, -1).mean(1)
        loss = criterion(output, target)
        # get the index of the max log-probability
#        pred = output.max(1, keepdim=True)[1]
#        correct += pred.eq(target.view_as(pred)).sum().item()
#        correct += (pred == target).sum()

        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), data.size(0))
        mapmeter.update(output, target)
        end = time.time()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            log_progress('Validation', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)

#    accuracy = 100. * correct / num_samples
#    accumeter.update(accuracy)
#    print(f'Valdiation Accuracy: {accuracy}')

    return loss_meter.avg, mapmeter.val


def main():
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Data loading
    loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
    val_loader = loaders.val_loader(imagetxt=args.valtxt)

    encoder = DenseNet121()
    model = TransferNet(encoder)

    if args.resume:
        if os.path.isfile(args.savefile):
            print("=> loading checkpoint '{}'".format(args.savefile))
            checkpoint = torch.load(args.savefile)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.savefile))
        else:
            print("=> no checkpoint found at '{}'".format(args.savefile))

    if args.parallel:
        model = nn.DataParallel(model)
        model = model.cuda()

    if args.cuda and not args.parallel:
        model.cuda()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    criterion = nn.BCEWithLogitsLoss(size_average=True)
    if args.cuda:
        criterion.cuda()

    val_meters = {
      'val_loss': AverageMeter(name='valloss'),
      'val_time': AverageMeter(name='valtime'),
      'val_mavep': mAPMeter(),
      'val_accuracy': AverageMeter(name='valaccuracy')
    }

    epoch_time = AverageMeter(name='epoch_time')
    end = time.time()
    print(f'Number of epochs: {args.num_epochs}')
    for epoch in range(1, args.num_epochs+1):
        print(f'epoch: {epoch}')
        val_loss, val_map = validate(val_loader, criterion, model, val_meters, args, epoch=epoch)
        epoch_time.update(time.time() - end)
        end = time.time()

    print(f"\nJob's done! Total runtime: {epoch_time.sum}, Average runtime: {epoch_time.avg}")
    val_meters['val_loss'].save('/home/ygx/lungs/lungs')
    val_meters['val_accuracy'].save('/home/ygx/lungs/lungs/acculogs')


if __name__=="__main__":
    main()
