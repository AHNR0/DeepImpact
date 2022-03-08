import numpy as np
import torch
from sklearn.metrics import confusion_matrix as CM

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter_confusion(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = np.array([[0.0,0.0],[0.0,0.0]])
        self.sum = np.array([[0.0,0.0],[0.0,0.0]])

    def update(self, val):
        self.val = val
        self.sum += val

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # print(maxk)
    batch_size = target.size(0)

    aaa, pred = output.topk(maxk, 1, True, True)
    # print(aaa)
    # print(pred)
    pred = pred.t()
    # print(pred)
    # print(target)
    # print(target.view(1,-1))
    # print(target.view(1, -1).expand_as(pred))
    # print(target.view(1,-1))
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)
    res = []
    # print(res)
    # print(topk)
    for k in topk:
        # print(correct[:k])
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        # print(res)
    return res


def accuracy_sens_prec(output, target):
    """Computes the precision@k for the specified values of k"""
    # print(output.size())
    # print(target.size())
    batch_size = target.size(0)
    _, preds = torch.max(output,1)
    confusion_matrix = torch.zeros(2, 2)
    for t, p in zip(target.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    # print(confusion_matrix)
    # confusion_matrix=CM(target.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy())
    # confusion_matrix= confusion_matrix.astype(float)
    confusion_matrix[0,1] += 0.001
    confusion_matrix[1,0] += 0.001
    # print(confusion_matrix)

    sensitivity=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])*100
    precision=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])*100
    accuracy=(confusion_matrix[1,1]+confusion_matrix[0,0])/(confusion_matrix[0,0]+confusion_matrix[1,0]+confusion_matrix[1,1]+confusion_matrix[0,1])*100
    return sensitivity, precision, accuracy, confusion_matrix