import torch
import numpy as np

from torchnet.meter import meter
from torchnet.meter import APMeter
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    """
    Computes and stores the average and current value

    Parameters:
    ----------
    name : str
        Name of the object to be tracked.

    rank : int
        Rank of the values computed (if distributed).
    """
    def __init__(self, name, rank=0):
        self.name = name
        self.rank = rank
        self.reset()

    def __str__(self):
        return f'Average Meter for {self.name} on rank {self.rank}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = np.empty(0)
        self.avgs = np.empty(0)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals = np.append(self.vals, self.val)
        self.avgs = np.append(self.avgs, self.avg)

    def save(self, path):
        np.save(path + '/' + self.name + '_avgs' + str(self.rank), self.avgs)
        np.save(path + '/' + self.name + '_vals' + str(self.rank), self.vals)


class AUCMeter(meter.Meter):
    """
    Lightly modified AUCMeter from pytorch.tnt

    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.
    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """

    def __init__(self, name, rank=None):
        super(AUCMeter, self).__init__()
        self.name = name
        self.rank = rank
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()
        self.tpr = 0.0
        self.fpr = 0.0
        self.area = 0.0

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().detach().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)

    def update(self):
        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5

        # sorting the arrays
        scores, sortind = torch.sort(torch.from_numpy(
            self.scores), dim=0, descending=True)
        scores = scores.numpy()
        sortind = sortind.numpy()

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        self.tpr /= (self.targets.sum() * 1.0)
        self.fpr /= ((self.targets - 1.0).sum() * -1.0)

        # calculating area under curve using trapezoidal rule
        n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        self.area = (sum_h * tpr).sum() / 2.0


class mAPMeter(meter.Meter):
    """
    The mAPMeter measures the mean average precision over all classes.
    The mAPMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """
    def __init__(self):
        super(mAPMeter, self).__init__()
        self.apmeter = APMeter()

    def reset(self):
        self.apmeter.reset()
        self.val = 0

    def update(self, output, target, weight=None):
        output = output.detach()
        target = target.detach()
        self.apmeter.add(output, target, weight)
        self.val = self.apmeter.value().mean()


def compute_auc(y_true, y_pred):
    """Computes Area Under the Curve (AUC) from output scores.

    :param output: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    :param target: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
    :return: list of AUROCs for each class.
    """
    # return np.random.rand(14,output.size(1))
    # global args
    aucs = []
    # print('target',target.type())
    # print('output:',output.type())
    if isinstance(y_true,torch.Tensor):
        y_true_np = y_true.cpu().data.numpy()
        y_pred_np = y_pred.cpu().data.numpy()
    else:
        y_true_np = y_true
        y_pred_np = y_pred

    for i in range(y_true_np.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true_np[:, i], y_pred_np[:, i]))
        except ValueError:
            # print('info: insufficient samples to compute AUC... ')
            aucs.append(0)

    return np.array(aucs)


class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self, name,keep=False):
        self.name=name
        self.keep=keep
        self.sum = 0
        self.n = 0
        self.val = 0
        self.store=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.n += n
        if self.keep:
            self.store.append(self.val)

    @property
    def avg(self):
        return self.sum / self.n

    @property
    def cum(self):
        return self.sum

    @property
    def data(self):
        return np.array(self.store)
