import torch.nn.functional as F

from .metrics import Metric


class Loss(Metric):
    def __init__(self):
        super().__init__()

    def _calculate_loss(self, output, batch):
        raise NotImplementedError

    def reset(self):
        self._sum = 0
        self._n = 0

    def update(self, output, batch):
        loss, loss_sum, n = self._calculate_loss(output, batch)
        self._sum += loss_sum
        self._n += n

        return loss

    @property
    def value(self):
        return self._sum / self._n


class CrossEntropyLoss(Loss):
    def __init__(self, device, input_key, target_key, weight=None, ignore_index=-100,
                 reduction='mean'):
        if reduction == 'none':
            raise ValueError('CrossEntropy: reduction can\'t be none')

        self._device = device
        self._input_key = input_key
        self._target_key = target_key
        self._weight = weight
        self._ignore_index = ignore_index
        self._reduction = reduction
        super().__init__()

    def _set_name(self):
        self.name = 'CrossEntropy({})'.format(self._target_key)

    def _calculate_loss(self, output, batch):
        _input = output[self._input_key]
        target = batch[self._target_key].to(device=self._device)
        loss = F.cross_entropy(
            _input, target, weight=self._weight, ignore_index=self._ignore_index,
            reduction=self._reduction)
        n = (target != self._ignore_index).sum().item()
        loss_sum = loss.item() * (n if self._reduction == 'mean' else 1)

        return loss, loss_sum, n
