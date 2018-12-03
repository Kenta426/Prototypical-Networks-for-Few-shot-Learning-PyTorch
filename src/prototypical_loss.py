# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import torch.nn as nn


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support, teacher_targets=None):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    if teacher_targets != None:
        teacher_targets_cpu = teacher_targets.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)


    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds1 = target_inds.contiguous().view(n_classes*n_query)
    #print(target_inds1)
    #loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    criterion = nn.CrossEntropyLoss()
    #Added soft labeling in CE and kd loss
    if teacher_targets == None:
        loss_val = criterion(-dists, target_inds1)
    else:
        #loss_val = criterion(-dists, target_inds1) + cross_entropy_soft(-dists, teacher_targets_cpu.view(n_classes*n_query,n_classes))
        alpha = .9
        loss_val = alpha*KL_loss(log_p_y, target_inds1) + (1-alpha)*KL_loss(log_p_y, teacher_targets_cpu.view(n_classes*n_query,n_classes))
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val


def get_prob(input, target, n_support):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    # print(dists.shape)

    # log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    #
    # target_inds = torch.arange(0, n_classes)
    # target_inds = target_inds.view(n_classes, 1, 1)
    # target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #
    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1)
    p_y = F.softmax(-dists, dim=1).view(n_classes, n_query, -1)
    return p_y

def cross_entropy_soft(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    #logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * torch.log(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * torch.log(input), dim=1))

    def KL_loss(x, y):
        torch.log(x)
        output = F.kl_div(torch.log(x), y, size_average = False)
        return output/x.size(0)


