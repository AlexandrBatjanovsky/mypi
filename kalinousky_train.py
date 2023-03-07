#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:27:26 2023

@author: alexandersn
"""

import time
import logging
from pathlib import Path

import json

from kalinousky_pipeline import DeepHDPipeline

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main_train():
    logging.basicConfig(filename= str(Path(__file__).with_suffix(".log").name), filemode='w', level = logging.INFO, force=True,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    cfg = json.load(open("kalinousky.json", "r"))
    pipeline = DeepHDPipeline(cfg, num_workers=2)
    callback = ModelCheckpoint(dirpath = pipeline.path_model, verbose=True, monitor='val_loss', mode = 'min')
    logger = TensorBoardLogger(save_dir = pipeline.path_model, name = '', version = 0)
    t1 = time.time()
    trainer = Trainer(accelerator = "gpu", callbacks=[callback],
                      max_epochs=pipeline.cfg["param"]["epochs"], logger=logger)
    if cfg["runs"]["checkpoint"]:
        trainer.fit(pipeline, ckpt_path=cfg["runs"]["checkpoint"])
    else:
        trainer.fit(pipeline)
    dt = time.time() - t1
    logging.info(f'\t\t... done, dt ~ {dt:0.2f} (s)')
    
if __name__ == '__main__':
    main_train()