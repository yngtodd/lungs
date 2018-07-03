import torch
import torch.nn as nn
from parser import parse_args
from data.hpml_loaders import XRayLoaders
import numpy as np
args = parse_args()

loaders = XRayLoaders(data_dir=args.data_dev, batch_size=1)
train_loader = loaders.train_loader(imagetxt=args.traintxt_dev,shuffle=False,transform=False)
number=[]
numbers=0
for i , (b,c) in enumerate(train_loader):
    print(type(b))
    print(type(c))
    print('---------')
    '''
    print(b.size())
    if b.size()[1]==4:
        number.append(i)
        numbers+=1
        print("count",numbers,"file",i)
    else: continue
    '''
#np.save("test_list.npy",number)
'''
imagetxt = args.traintxt_dev
train = np.load("list_num.npy")
val = np.load("val_list.npy")
file_names=[]
with open(imagetxt, "r") as f:
    i=0
    for line in f:
        if i in train:
            items = line.split()
            image_name= str(items[0])
            file_names.append(image_name)
            i+=1
        else:
            i+=1

imagetxt = args.valtxt_dev
with open(imagetxt, "r") as f:
    i=0
    for line in f:
        if i in val:
            items = line.split()
            image_name= str(items[0])
            file_names.append(image_name)
            i+=1
        else:
            i+=1

np.save("file_names.npy",file_names)
print("length of filenames",np.shape(file_names))
'''

