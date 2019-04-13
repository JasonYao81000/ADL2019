import csv
from itertools import product

import torch
from tqdm import tqdm

from .utils import FixedOrderedDict


class Stat:
    def __init__(self, losses, metrics, log_path):
        self._fieldnames = [loss.name for loss in losses] + \
            (['total_loss'] if len(losses) > 1 else []) + \
            [metric.name for metric in metrics]
        self.reset()

        log_fieldnames = ['{}_{}'.format(mode, name) for mode, name
                          in product(['train', 'eval'], self._fieldnames)]
        self._log_writer = csv.DictWriter(
            log_path.open(mode='w', buffering=1), fieldnames=log_fieldnames)
        self._log_writer.writeheader()

    def reset(self):
        self._stat = {
            'train': FixedOrderedDict({name: 0 for name in self._fieldnames}),
            'eval': FixedOrderedDict({name: 0 for name in self._fieldnames})
        }

    def __getitem__(self, key):
        if key not in ['train', 'eval']:
            raise KeyError('Stat: Key must be one of {train, eval}')
        return self._stat[key]

    def log(self):
        self._log_writer.writerow(
            {'{}_{}'.format(mode, name): self._stat[mode][name]
             for mode, name in product(['train', 'eval'], self._fieldnames)})

    @property
    def stat(self):
        return {
            'train': self._stat['train'].get_dict(),
            'eval': self._stat['eval'].get_dict()
        }


class BaseTrainer:
    def __init__(self, device, cfg, train_data_loader, dev_data_loader, model, losses,
                 metrics, log_path, ckpt_dir):
        self._device = device
        self._cfg = cfg
        self._train_data_loader = train_data_loader
        self._dev_data_loader = dev_data_loader
        self._model = model
        self._losses = losses
        self._metrics = metrics
        self._stat = Stat(losses, metrics, log_path)
        self._ckpt_dir = ckpt_dir

    def start(self):
        tqdm.write('[-] Start training!')
        bar = tqdm(
            range(self._cfg.n_epochs), desc='[Total progress]', leave=False,
            position=0, dynamic_ncols=True)
        for epoch in bar:
            self._stat.reset()
            self._epoch = epoch + 1
            self._run_epoch('train')
            self._run_epoch('eval')
            self._stat.log()
            self._save_ckpt()
        tqdm.write('[-] Training done!')
        bar.close()

    def _run_epoch(self, mode):
        if mode == 'train':
            data_loader = self._train_data_loader
            self._model.set_train()
            desc_prefix = 'Train'
        elif mode == 'eval':
            data_loader = self._dev_data_loader
            self._model.set_eval()
            desc_prefix = 'Eval '
        torch.set_grad_enabled(mode == 'train')
        self._reset_losses_and_metrics()
        self._model.zero_grad()

        bar = tqdm(
            data_loader, desc='[{} epoch {:2}]'.format(desc_prefix, self._epoch),
            leave=False, position=1, dynamic_ncols=True)
        for idx, batch in enumerate(bar):
            output = self._run_batch(batch)
            loss = self._calculate_losses(mode, output, batch)
            if mode == 'train':
                loss.backward()
                is_acc_step = (idx + 1) % self._cfg.n_gradient_accumulation_steps == 0
                is_final_step = idx == len(data_loader) - 1
                if is_acc_step or is_final_step:
                    if self._cfg.max_grad_norm > 0:
                        self._model.clip_grad(self._cfg.max_grad_norm)
                    self._model.update()
                    self._model.zero_grad()
            self._calculate_metrics(mode, output, batch)
            bar.set_postfix_str(str(self._stat[mode]))
        bar.close()

    def _reset_losses_and_metrics(self):
        for loss in self._losses:
            loss.reset()
        for metric in self._metrics:
            metric.reset()

    def _run_batch(self, batch):
        raise NotImplementedError

    def _calculate_losses(self, mode, batch, output):
        total_loss = 0
        for loss in self._losses:
            total_loss += loss.update(batch, output)
            self._stat[mode][loss.name] = loss.value
        if len(self._losses) > 1:
            self._stat[mode]['total_loss'] = total_loss.item()

        return total_loss

    def _calculate_metrics(self, mode, batch, output):
        for metric in self._metrics:
            metric.update(batch, output)
            self._stat[mode][metric.name] = metric.value

    def _save_ckpt(self):
        self._model.save_state(self._epoch, self._stat.stat, self._ckpt_dir)
