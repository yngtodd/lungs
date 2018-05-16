import numpy as np
import pickle
import os

# import matplotlib.pyplot as plt

# _path = os.path.join(os.getcwd(),'AIM-1','data','uq')
_path = ''
_class_names = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def save_pkl(fdir=None, f=None, obj=None):
    global _path
    if _path is None:
        setpath()
        print('setting default path: ...%s' %(_path))
    if fdir is None:
        fdir = ''
    path = os.path.join(_path,fdir)
    saveas = os.path.join(path, str(f) + '.pkl')
    #print(saveas, '... saving ...')
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            raise OSError('can\'t create directory %s' % path)
    elif os.path.isfile(saveas):
        try:
            os.remove(saveas)
        except:
            raise OSError('can\'t remove file %s' % saveas)

    with open(saveas, mode='wb') as output:
        pickler = pickle.Pickler(output, -1)
        pickler.dump(obj)
        output.close()


def load_pkl(fdir=None, f=None):
    global _path
    if _path is None:
        setpath()
        print('setting default path: ...%s' %(_path))
    if fdir is None:
        fdir = ''
    # saveas = dir + str(f) + '.pkl'
    path = os.path.join(_path,fdir)
    loadas = os.path.join(path, str(f) + '.pkl')
    #print(loadas,'... loading ....')

    if os.path.isfile(loadas):
        try:
            input = open(loadas, mode='rb')
            dat = pickle.load(input)
            input.close()
            return dat
        except:
            raise OSError('can\'t open file %s' % loadas)

    return None


def setpath(path=os.path.join(os.getcwd(),'lungXnet','data')):
    global _path
    _path=path
