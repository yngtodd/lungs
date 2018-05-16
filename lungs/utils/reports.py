import numpy as np
import matplotlib.pyplot as plt
import utils.utils as util
from sklearn.metrics import roc_auc_score

_path='/home/fa6/data/scp/trial_2018050816_full/'
util.setpath(_path)
_n_classes = 14
_class_names = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


train_loss=util.load(f='train_loss')
val_loss=util.load(f='val_loss')
train_auc=util.load(f='train_auc')
val_auc=util.load(f='val_auc')
test_results = util.load(f='test_results')

test_results['auc'] = []
for i in range(_n_classes):
    test_results['auc'].append(roc_auc_score(y_true=test_results['y_true'][:, i],y_score=test_results['y_pred'][:, i]))

test_results['auc'] = np.array(test_results['auc'])

plt.figure
plt.plot(np.arange(len(val_auc['data'])),val_auc['data'],color='r')
plt.plot(np.arange(len(train_auc['data'])),train_auc['data'],color='b')
plt.xlabel('epoch')
plt.ylabel('auc')
# plt.legend(['val','train','test'])
plt.legend(['val','train'])
plt.title('performance')

plt.figure()
plt.plot(np.arange(len(val_loss['data'])),val_loss['data'],color='r')
plt.plot(np.arange(len(train_loss['data'])),train_loss['data'],color='b')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.legend(['val','train','test'])
plt.legend(['val','train'])
plt.title('loss')


print('avg AUC is {AUROC_avg:.3f}'.format(AUROC_avg=test_results['auc'].mean()))
for i in range(_n_classes):
    print('AUC for {_class_names}:\t{test:.3f}'.format(
         _class_names=_class_names[i],
          test=test_results['auc'][i]))


