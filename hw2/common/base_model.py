from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm


class BaseModel:
    def __init__(self, device, *args, **kwargs):
        self._device = device
        self._net, self._optim = self._create_net_and_optim(*args, **kwargs)

    def _create_net_and_optim(self, *args, **kwargs):
        raise NotImplementedError

    def set_train(self):
        self._net.train()

    def set_eval(self):
        self._net.eval()

    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)

    @property
    def parameters(self):
        return self._net.parameters()

    def zero_grad(self):
        self._optim.zero_grad()

    def clip_grad(self, max_grad_norm):
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self._net.parameters()), max_grad_norm)

    def update(self):
        self._optim.step()

    def save_state(self, epoch, stat, ckpt_dir):
        tqdm.write('[*] Saving model state')
        ckpt_path = ckpt_dir / 'epoch-{}.ckpt'.format(epoch)
        torch.save({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'stat': stat,
            'net_state': self._net.state_dict(),
            'optim_state': self._optim.state_dict()
        }, ckpt_path)
        tqdm.write('[-] Model state saved to {}\n'.format(ckpt_path))

    def load_state(self, ckpt_path):
        print('[*] Loading model state')
        ckpt = torch.load(ckpt_path)
        self._net.load_state_dict(ckpt['net_state'])
        self._net.to(device=self._device)
        self._optim.load_state_dict(ckpt['optim_state'])
        for state in self._optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self._device)
        # print('[-] Model state loaded from {}\n'.format(ckpt_path))
