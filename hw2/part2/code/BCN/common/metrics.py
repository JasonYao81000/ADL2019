import torch.nn.functional as F


class Metric:
    def __init__(self):
        self._set_name()
        self.reset()

    def _set_name(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def update(self, output, batch):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError


class Accuracy(Metric):
    def __init__(self, device, key):
        self._device = device
        self._key = key
        super().__init__()

    def _set_name(self):
        self.name = 'acc({})'.format(self._key)

    def reset(self):
        self._sum = 0
        self._n = 0

    def update(self, output, batch):
        prediction = output[self._key].detach()
        target = batch[self._key].to(device=self._device)
        self._sum += (prediction == target).sum().item()
        self._n += len(prediction)

    @property
    def value(self):
        return self._sum / self._n
