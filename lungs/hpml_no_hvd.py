import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from parser import parse_args
#from data.hpml_loaders import XRayLoaders
from data.hpml_loaders import XRayLoaders
from models.lungXnet import LungXnet
import torch.utils.data.distributed
import time
import pickle

def train(epoch, train_loader, optimizer, criterion, model,dictionary, args):
    """"""
    #loss_meter = meters['train_loss']
    #batch_time = meters['train_time']
    #mapmeter = meters['train_mavep'] 
    num_samples = len(train_loader)

    model.train()
    #train_loss = Metric('train_loss')
    end = time.time()
    #print(f'Args.fp16 is {args.fp16}')
    for batch_idx, (data, target) in enumerate(train_loader):
        #print("data shape",data.size())
        #bs,n_crops,c, h, w = data.size()
        bs,c,h,w = data.size()
        data= data.view(-1,c,h,w)
        #data = data.permute(1,0,2,3)
        if args.cuda:
            data = data.contiguous()
            data = data.cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        #output = output.view(bs,n_crops, -1).mean(1)
        #print("output size", output.size(), "target size", target.size())
        #assert (output.data >= 0. & output.data <= 1.).all()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #train_loss.update(loss)
        #batch_time.update(time.time() - end)
        #loss_meter.update(loss.item(), data.size(0))
        #mapmeter.update(output, target)
        #end = time.time()
    #dictionary[epoch]=train_loss.avg.item()
    dictionary[epoch]=loss
    print("time per epoch",time.time()-end)
        #if batch_idx % args.log_interval == 0 and batch_idx > 0:
            #log_progress('Train', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)
    #print("time taken for training epoch",time.time()-end)   

def validate(epoch, val_loader, criterion, model,dictionary, args):
    """"""
    #loss_meter = meters['val_loss']
    #batch_time = meters['val_time']
    #mapmeter = meters['val_mavep']
    num_samples = len(val_loader)

    model.eval()
    #val_loss = Metric('val_loss')
    end = time.time()
    for batch_idx, (data, target) in enumerate(val_loader):
        bs, c, h, w = data.size()
        data = data.view(-1,c,h,w)
        #data = data.permute(1,0,2,3)
        if args.cuda:
            data = data.contiguous()
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = model(data)
        #print(f'output has size {output.size()}')
        loss = criterion(output, target)
        #val_loss.update(loss)
        #batch_time.update(time.time() - end) 
        #loss_meter.update(loss.item(), data.size(0))
        #mapmeter.update(output, target)
        #end = time.time()
    dictionary[epoch] =loss
        #if batch_idx % args.log_interval == 0 and batch_idx > 0:
            #log_progress('Validation', epoch, args.num_epochs, batch_idx, num_samples, batch_time, loss_meter, mapmeter)
    #print("time taken for validation epoch",time.time()-end)
    
def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if args.fp16:
        assert torch.backends.cudnn.enabled
    data_time = time.time()	
    # Data loadingi
    if args.summitdev:
        loaders = XRayLoaders(data_dir=args.data_dev, batch_size=args.batch_size)
        train_loader = loaders.train_loader(imagetxt=args.traintxt_dev)
        #val_loader = loaders.val_loader(imagetxt=args.valtxt_dev)
    else:
        loaders = XRayLoaders(data_dir=args.data, batch_size=args.batch_size)
        train_loader = loaders.train_loader(imagetxt=args.traintxt)
        #val_loader = loaders.val_loader(imagetxt=args.valtxt)
        print("data loaded for Summit in time",time.time()-data_time)


    model = LungXnet()
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break	
    if args.cuda and args.parallel:
        model = nn.DataParallel(model)
        model = model.cuda()
        print("model loaded in parallel"_
    else:
        model.cuda()
        print("model loaded in serial")
	
    #print(model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
	#Horovod Optimizer
    #optimizer = hvd.DistributedOptimizer(
    #	    optimizer, named_parameters=model.named_parameters())
	
    
    criterion = nn.BCEWithLogitsLoss()
    criterion.cuda()
    if resume_from_epoch>0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])  
    
    start = time.time()
    #val_loss_dict = dict()

    def save_checkpoint(epoch):
        torch.save(model.state_dict(), args.checkpoint_format.format(epoch=epoch + 1))

    for epoch in range(resume_from_epoch, resume_from_epoch+args.num_epochs):
        train(epoch, train_loader, optimizer, criterion, model,train_loss_dict, args)
        validate(epoch, val_loader, criterion, model,val_loss_dict, args)
        save_checkpoint(epoch)
    print("\nJob's done! Total runtime:",time.time()-start)
    with open('train_loss.pickle', 'wb') as handle:
        pickle.dump(train_loss_dict, handle)
    #with open('val_loss.pickle', 'wb') as handle:
    #    pickle.dump(val_loss_dict, handle)
    
if __name__=="__main__":
    main()
