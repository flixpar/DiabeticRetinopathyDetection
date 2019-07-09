import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, label_input, label_target):
        """Compute focal loss for multi-class problem.
        Ignores anchors having -1 target label
        """
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = label_target != self.ignore

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha)
        return loss

def sigmoid_focal_loss(input: torch.Tensor,
                       target: torch.Tensor,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    """Compute binary focal loss between target and output logits.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
    References::
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)

    return loss
