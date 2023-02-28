#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:27:26 2023

@author: alexandersn
"""

import time
import logging

import json

from kalinousky_pipeline import DeepHDPipeline

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main_train():
    logging.basicConfig(level=logging.INFO)
    cfg = json.load(open("kalinousky.json", "r"))
    pipeline = DeepHDPipeline(cfg, num_workers=2)
    callback = ModelCheckpoint(dirpath = pipeline.path_model, verbose=True, monitor='val_loss', mode = 'min')
    logger = TensorBoardLogger(save_dir = pipeline.path_model, name = '', version = 0)
    t1 = time.time()
    trainer = Trainer(accelerator = "gpu", callbacks=[callback],
                      max_epochs=pipeline.cfg['epochs'], logger=logger)
    #distributed_backend='ddp')
    if cfg["runs"]["checkpoint"]:
        trainer.fit(pipeline, ckpt_path=cfg["runs"]["checkpoint"])
    else:
        trainer.fit(pipeline)
    dt = time.time() - t1
    logging.info(f'\t\t... done, dt ~ {dt:0.2f} (s)')
    
if __name__ == '__main__':
    main_train()