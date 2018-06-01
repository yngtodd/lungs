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
    parser.add_argument('--traintxt', type=str, default='/mnt/data/ChestXRay14/train_list.txt',
                        help='path to training set text info (image names + labelss')
    parser.add_argument('--valtxt', type=str, default='/mnt/data/ChestXRay14/val_list.txt',
                        help='path to validation set text info')
    parser.add_argument('--texttxt', type=str, default='/mnt/data/ChestXRay14/test_list.txt',
                        help='path to test set text info')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--num_epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='S',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 2)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
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
    parser.add_argument('--log_interval', '-p', default=2, type=int,
                        metavar='N', help='print frequency (default: 2)')
    parser.add_argument('--resume', default='checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: checkpoint.pth.tar)')
    parser.add_argument('--best-model', default='model.pth.tar', type=str, metavar='PATH',
                        help='save best model (default: model.pth.tar)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-f','--factor',default=0.1,type=float,
                        metavar='F',help='learning rate sheduler factor (default: .1)')
    parser.add_argument('--patience',default=5,type=int,
                        metavar='P',help='learning rate sheduler patience (default: 5)')
    parser.add_argument('--trial', dest='trial', action='store_true',
                        help='trial experiment with smaller dataset')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables cuda training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for experiments. [default: 42]')
    parser.add_argument('--fp16', default = True,  action='store_true',
                        help='running model in half precision')
    args = parser.parse_args()
    return args
