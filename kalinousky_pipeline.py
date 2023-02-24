#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:21:38 2023

@author: alexandersn
"""

import os
import time

import numpy as np

import logging

import torch
from torch.utils.data import DataLoader
from deephd_losses import build_loss_by_name
from kalinousky_model import ASPPResNetSE
#from kalinousky_data import ProtContactDataSet
from pytorch_lightning import LightningModule
import pytorch_lightning as pl



def _get_random_seed():
    seed = int(time.time() * 100000) % 10000000 + os.getpid()
    return seed


def worker_init_fn_random(idx):
    seed_ = _get_random_seed() + idx
    torch.manual_seed(seed_)
    np.random.seed(seed_)


class DeepHDPipeline(LightningModule):

    def __init__(self, path_cfg: str, num_workers=8):
        super().__init__()
        self.path_cfg = path_cfg
        self.num_workers = num_workers
        self.cfg = load_config(self.path_cfg)
        self.model_prefix = self.cfg['model']['type']
        self.model = ASPPResNetSE(inp=43, out=1,
                                  nlin=self.cfg['model']['nlin'],
                                  num_stages=self.cfg['model']['num_stages'])
        self.trn_loss = build_loss_by_name(self.cfg['loss'])
        self.val_losses = {}
        self.val_losses['val_loss'] = build_loss_by_name(self.cfg['loss'])
        for x in self.cfg['val_losses']:
            self.val_losses[f'val_loss_{x}'] = build_loss_by_name(x)

    def _get_model_prefix(self) -> str:
        ret = '{}_s{}_{}'.format(self.cfg['model']['type'],
                                 self.cfg['model']['num_stages'],
                                 self.cfg['model']['nlin'])
        return ret

    def build(self, model_dir='models_last1'):

        # path_trn = self.cfg['trn_abs']
        # path_val = self.cfg['val_abs']
        # self.dataset_trn = DHDDataset(path_idx=self.cfg['trn_abs'], crop_size=self.cfg['crop_size'],
        #                               params_aug=self.cfg['aug'], test_mode=False, num_fake_iters=self.cfg['iter_per_epoch']).build()
        # self.dataset_val = DHDDataset(path_idx=self.cfg['val_abs'], crop_size=self.cfg['crop_size'],
        #                               params_aug=None, test_mode=True).build()
        model_prefix = self._get_model_prefix()
        self.path_model = os.path.join(self.cfg['wdir'], model_dir,
                                       os.path.basename(self.path_cfg) + '_model_' + model_prefix + '_l{}'.format(self.cfg['loss']))
        logging.info(f'Pipeline:\n\nmodel = {self.model}\n\tloss-train = {self.trn_loss}\n\tloss-val = {self.val_losses}')
        return self

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x = batch['inp']
        y_gt = batch['out'].type(torch.float32)
        y_pr = self.model(x)
        loss = self.trn_loss(y_pr, y_gt)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        # return {'loss': tensorboard_logs}

    # def training_end(self, outputs):
    #     keys_ = list(outputs[0].keys())
    #     # ret_avg = {f'{k}': torch.stack([x[k] for x in outputs]).mean() for k in keys_}
    #     k = 'train_loss'
    #     ret_avg = {k: torch.stack([x[k] for x in outputs]).mean()}
    #     # ret_avg =
    #     tensorboard_logs = ret_avg
    #     ret = {'log': tensorboard_logs}
    #     return ret

    def validation_step(self, batch, batch_nb):
        x = batch['inp']
        y_gt = batch['out'].type(torch.float32)
        y_pr = self.model(x)
        ret = {loss_name: loss_func(y_pr, y_gt) for loss_name, loss_func in self.val_losses.items()}
        return ret

    def validation_end(self, outputs):
        keys_ = list(outputs[0].keys())
        ret_avg = {f'{k}': torch.stack([x[k] for x in outputs]).mean() for k in keys_}
        tensorboard_logs = ret_avg
        ret = {'log': tensorboard_logs}
        return ret

    def configure_optimizers(self):
        ret = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return ret

    @pl.data_loader
    def train_dataloader(self):
        ret = DataLoader(self.dataset_trn, num_workers=self.num_workers,
                         batch_size=self.cfg['batch_size'],
                         worker_init_fn=worker_init_fn_random)  # , collate_fn=skip_none_collate)
        return ret

    @pl.data_loader
    def val_dataloader(self):
        ret = DataLoader(self.dataset_val, num_workers=self.num_workers,
                         batch_size=self.cfg['batch_size_val'],
                         worker_init_fn=worker_init_fn_random)  # , collate_fn=skip_none_collate)
        return ret