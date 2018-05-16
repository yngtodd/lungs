# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import time
import copy
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import shutil


_path = '/mnt/data'
# _path = '/home/fa6/data/'
CKPT_PATH = './data/checkpoint.pth.tar'
MODEL_PATH = './data/model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = os.path.join(_path,'ChestXRay14/images')
# DATA_DIR = os.path.join(_path,'chestX-ray14/images')
# TEST_IMAGE_LIST = './chestX-ray14/labels/test_list.txt'
# TRAIN_IMAGE_LIST = './chestX-ray14/labels/train_list.txt'
IMAGE_LIST_FILES = {'train':'./chestX-ray14/labels/train_list.txt',
                    'test':'./chestX-ray14/labels/test_list.txt',
                    'val':'./chestX-ray14/labels/val_list.txt'}
BATCH_SIZE = 16
use_gpu = torch.cuda.is_available()
resume=False
isinit=False

def init():
    global isinit, dataloaders, dataset_sizes
    if isinit:
        return
    else:
        isinit = True

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
    }

    image_datasets = {x:ChestXrayDataSet(data_dir=DATA_DIR,
                                        image_list_file=IMAGE_LIST_FILES[x],
                                        transform=data_transforms[x])
                     for x in ['train','test','val']}

    dataloaders = {x:DataLoader(dataset=image_datasets[x], batch_size=BATCH_SIZE,
                              shuffle=False if x =='test' else True, num_workers=8, pin_memory=True)
                   for x in ['train','test','val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','test', 'val']}


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

#train model
def train(model, criterion,optimizer, scheduler, num_epochs=100):
    global dataloaders,dataset_sizes
    init()

    # #---delete before commit---#
    # num_epochs=2
    # #--------------------------#
    t1 = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print(('-'*10))
        is_best = False
        for phase in ['train','val']:
            t_epoch = time.time()
            istrain = phase=='train'
            model.train(istrain)
            if istrain:
                scheduler.step()

            running_loss = 0.0
            gts,preds = torch.FloatTensor().cuda(),torch.FloatTensor().cuda()
            # gts,preds = gts,preds.cuda()
            batch_no=0
            for (inputs,y_trues) in dataloaders[phase]:
                if not batch_no%300:
                    print('batch {}'.format(batch_no))
                batch_no +=1
                if not istrain:
                    bs, n_crops, c, h, w = inputs.size()
                    if use_gpu:
                        inputs = Variable(inputs.view(-1, c, h, w).cuda(), volatile=True)
                        y_trues = Variable(y_trues.cuda())
                    else:
                        inputs = Variable(inputs.view(-1, c, h, w), volatile=True)
                        y_trues = Variable(y_trues)
                else:
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        y_trues = Variable(y_trues.cuda())
                    else:
                        inputs = Variable(inputs)
                        y_trues = Variable(y_trues)

                optimizer.zero_grad()

                outputs = model(inputs)

                if istrain:
                    loss = criterion(outputs,y_trues)
                else:
                    loss = criterion(outputs.view(bs, n_crops, -1).mean(1), y_trues)

                if istrain:
                    loss.backward()
                    optimizer.step()


                running_loss += loss.data[0] * inputs.size(0)
                if istrain:

                    preds = torch.cat((preds,outputs.data), dim=0)
                else:
                    preds = torch.cat((preds,outputs.view(bs, n_crops, -1).mean(1).data), dim=0)
                gts = torch.cat((gts,y_trues.data),dim=0)

            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_auc = np.array(compute_AUCs(gt=gts,pred=preds)).mean()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_auc))
            if not istrain and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                is_best = True
            t_epoch_elapsed = time.time() - t_epoch
            print('epoch complete in {:.0f}m {:.0f}s'.format(
                t_epoch_elapsed // 60, t_epoch_elapsed % 60))

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    time_elapsed = time.time() - t1
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model=None):
    # initialize the ground truth and output tensor
    global dataloaders, dataset_sizes, isinit
    init()
    if model is None:
        cudnn.benchmark = True

        model = DenseNet121(N_CLASSES).cuda()
        model = torch.nn.DataParallel(model).cuda()

        if os.path.isfile(MODEL_PATH):
            print("=> loading model")
            checkpoint = torch.load(MODEL_PATH)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded model")
        else:
            print("=> no model found")
            print("first train model")
            return

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode

    model.eval()

    for i, (inp, target) in enumerate(dataloaders['test']):
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)
        if i%200 ==0:
            print('testing.... {} of {}'.format(i,dataset_sizes['test']))


    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def imshow(inp,title=None):
    """
    :param inp: Imshow for Tensor.
    :param title:
    :return:
    """
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([.485, .456, .406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def save_checkpoint(state, is_best):
    print('-'*5+'saving checkpoint'+'-'*5)
    torch.save(state, CKPT_PATH)
    if is_best:
        shutil.copyfile(CKPT_PATH, MODEL_PATH)


def main():
    init()
    global data_transforms,image_datasets,dataloaders,dataset_sizes
    nepochs=100
    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9)

    if os.path.isfile(MODEL_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        nepochs -= checkpoint['epoch']
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)
    criterion = nn.BCELoss(size_average = True)


    model = train(model, criterion, optimizer,scheduler,num_epochs=nepochs)

    # test_model(model)



if __name__ == '__main__':
    main()
