import argparse


def parse_args():
    """
    Parse Arguments for the lungXnet.
    
    Returns:
    -------
    * `args`: [argparse object]
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch lungXnet Training')
    parser.add_argument('-d','--data', metavar='DIR',default='/mnt/data/ChestXRay14/images',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet121',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: densenet121)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='S',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch', default=14, type=int,
                        metavar='N', help='mini-batch size (default: 14)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--step-size', '--step-size', default=30, type=int,
                        metavar='SS', help='learning rate scheduler step size')
    parser.add_argument('--gamma', default=.1, type=float,
                        metavar='G', help='learning rate scheduler gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIM',
                        help='type of optimizer (default=Adam)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=14, type=int,
                        metavar='N', help='print frequency (default: 21)')
    parser.add_argument('--resume', default='checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: checkpoint.pth.tar)')
    parser.add_argument('--best-model', default='model.pth.tar', type=str, metavar='PATH',
                        help='save best model (default: model.pth.tar)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-t','--tag',default='trial_{x:%Y%m%d%H}'.format(x=datetime.datetime.now()),type=str,
                        help='unique tag for storing experiment data')
    parser.add_argument('-f','--factor',default=0.1,type=float,
                        metavar='F',help='learning rate sheduler factor (default: .1)')
    parser.add_argument('--patience',default=5,type=int,
                        metavar='P',help='learning rate sheduler patience (default: 5)')
    parser.add_argument('--trial', dest='trial', action='store_true',
                        help='trial experiment with smaller dataset')
    args = parser.parse_args()
    return args
