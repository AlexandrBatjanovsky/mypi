#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:21:38 2023

@author: alexandersn
"""

import time
import os
from pathlib import Path
import logging

import numpy

import torch
from torch.utils.data import DataLoader
from kalinousky_data import ProtContactDataSet
from pytorch_lightning import LightningModule
from segmentation_models_pytorch.utils.losses import DiceLoss, JaccardLoss

from kalinousky_model import ASPPResNetSE

def _get_random_seed():
    seed = int(time.time() * 100000) % 10000000 + os.getpid()
    return seed


def worker_init_fn_random(idx):
    seed_ = _get_random_seed() + idx
    torch.manual_seed(seed_)
    numpy.random.seed(seed_)

def build_loss_by_name(loss_name: str) -> torch.nn.Module:
    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name == "l1":
        return torch.nn.L1Loss()
    elif loss_name == "l2":
        return torch.nn.MSELoss()
    # elif loss_name == "bcedice":
    #     return BCEDiceLoss()
    # elif loss_name == "bcejaccard":
    #     return BCEJaccardLoss()
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name == "jaccard":
        return JaccardLoss()
    else:
        raise NotImplementedError


#def _get_random_seed():
#    seed = int(time.time() * 100000) % 10000000 + os.getpid()
#    return seed

#def worker_init_fn_random(idx):
#    seed_ = _get_random_seed() + idx
#    torch.manual_seed(seed_)
#    np.random.seed(seed_)


class DeepHDPipeline(LightningModule):

    def __init__(self, conf: dict, num_workers=4):
        super().__init__()
        self.num_workers = num_workers
        self.cfg = conf
        self.model_prefix = self.cfg["model"]["type"]
        self.model = ASPPResNetSE(inp=43, out=1,
                                  nlin=self.cfg["model"]["nlin"],
                                  num_stages=self.cfg["model"]["num_stages"])
        self.trn_loss = build_loss_by_name(self.cfg["model"]["loss"])
        self.val_loss = build_loss_by_name(self.cfg["model"]["loss"])

        model_prefix = "{}_s{}_{}".format(self.cfg["model"]["type"],
                         self.cfg["model"]["num_stages"],
                         self.cfg["model"]["nlin"])

        self.path_model = Path(self.cfg["runs"]["wdir"], "models_lastl",
                               "_model_" + model_prefix + "_l{}".format(self.cfg["model"]["loss"]))
        logging.info(f"Pipeline:\n\nmodel = {self.model}\n\tloss-train = {self.trn_loss}")

        self.dataset_trn = ProtContactDataSet(dat_files_csv=self.cfg["data"]["tcs"], test_mode=False,
                                              num_iters=self.cfg["param"]["num_iters"])
        
        self.dataset_val = ProtContactDataSet(dat_files_csv=self.cfg["data"]["vcs"], test_mode=True)
        #return self 
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x = batch["inp"]
        y_gt = batch["out"].type(torch.float32)
        y_pr = self.model(x)
        loss = self.trn_loss(y_pr, y_gt)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # def training_end(self, outputs):
    #     keys_ = list(outputs[0].keys())
    #     # ret_avg = {f"{k}": torch.stack([x[k] for x in outputs]).mean() for k in keys_}
    #     k = "train_loss"
    #     ret_avg = {k: torch.stack([x[k] for x in outputs]).mean()}
    #     # ret_avg =
    #     tensorboard_logs = ret_avg
    #     ret = {"log": tensorboard_logs}
    #     return ret

    def validation_step(self, batch, batch_nb):
        x = batch["inp"]
        y_gt = batch["out"].type(torch.float32)
        y_pr = self.model(x)
        loss = self.val_loss(y_pr, y_gt)
        self.log("valid_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        ret = torch.optim.Adam(self.model.parameters(), lr=self.cfg["param"]["learn_rate"])
        return ret

    def train_dataloader(self):
        ret = DataLoader(self.dataset_trn, num_workers=self.num_workers,
                         batch_size=self.cfg["param"]["batch"], 
                         worker_init_fn=worker_init_fn_random)
        return ret

    def val_dataloader(self):
        ret = DataLoader(self.dataset_val, num_workers=self.num_workers,
                         batch_size=self.cfg["param"]["batch"],
                         worker_init_fn=worker_init_fn_random)
        return ret
