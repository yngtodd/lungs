import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lungs.parser import parse_args
from lungs.data.loaders import XRayLoaders
from lungs.models.lungxnet import LungXnet

from lungs.utils.logger import print_progress
from lungs.meters import AverageMeter, AUCMeter


def train(epoch, train_loader, optimizer, criterion):
    """"""


def validate(epoch, test_loader):
    """"""


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Data loading
    loaders = XRayLoaders(data_dir=args.data_dir, batch_size=args.batch_size)
    train_loader = loaders.train_loader(imagetxt=args.traintxt)
    val_loader = loaders.val_loader(imagetxt=args.valtxt)

    model = LungXnet()
    if args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
