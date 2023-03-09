#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:11:07 2023

@author: alexandersn
"""

import pandas
import numpy
import torch
from torch.utils.data import Dataset

import logging
from joblib import load
from pathlib import Path
import time
import sys

from prot_utils import AminsCode

import matplotlib.pyplot as plt

class ProtContactDataSet(Dataset):
    def __init__(self, dat_files_csv: str, crop = 0, test_mode = False, num_iters=100):
        self.dat_files_csv = dat_files_csv
        self.test_mode = test_mode
        self.num_iters = num_iters
        self.crop_size = crop
        
        # load csv with pair globuls
        self.CSVFileData = pandas.read_csv(self.dat_files_csv)
        self.CSVFileData.columns = ["struct", "filename", "models", "imodel", "chain1", "chain2"]
        self.data = []
        
        # load data of pair globuls
        for c_rec in self.CSVFileData.groupby("filename"):
            jbl_dat = load(c_rec[0])
            for index, row in c_rec[1].iterrows():
                self.data.append({
                                    "head": [row["struct"], row["imodel"], row["chain1"], row["chain2"]],
                                    "inp": {"seq1": jbl_dat[row["imodel"]]["Seqs"][row["chain1"]],
                                            "seq2": jbl_dat[row["imodel"]]["Seqs"][row["chain2"]],
                                            "in_dis1": jbl_dat[row["imodel"]]["InDists"][row["chain1"]],
                                            "in_dis2": jbl_dat[row["imodel"]]["InDists"][row["chain2"]]},
                                    "out": jbl_dat[row["imodel"]]["PPIs"][(row["chain1"], row["chain2"])]
                                })
                
    def __getitem__(self, item):
        # create sequence code layers (2*Aminotable layers)
        def AminsCodeLine_to_AminsCodeLayers(SeqCode: list):
            HorSeq, VerSeq = numpy.meshgrid(SeqCode, SeqCode)
            shp_inp = HorSeq.shape
            HorSeq = numpy.eye(len(AminsCode))[HorSeq.reshape(-1)]
            VerSeq = numpy.eye(len(AminsCode))[VerSeq.reshape(-1)]
            HorSeq = HorSeq.reshape(shp_inp + (len(AminsCode),))
            VerSeq = VerSeq.reshape(shp_inp + (len(AminsCode),))
            TensorSeq = numpy.dstack([HorSeq, VerSeq])
            return TensorSeq
        
        sample = self.data[item]
        #print(sample["inp"])
        # A1 and B1
        # [internal dis[SxS], 2*aminotable sequence code layers[SxS]]
        MSeg1 = numpy.dstack([sample["inp"]["in_dis1"][..., None], 
                              AminsCodeLine_to_AminsCodeLayers(sample["inp"]["seq1"]).astype(numpy.float32)])

        MSeg2 = numpy.dstack([sample["inp"]["in_dis2"][..., None], 
                              AminsCodeLine_to_AminsCodeLayers(sample["inp"]["seq2"]).astype(numpy.float32)])

        # create matrix
        # [A1, 0]
        # [0, B1]
        MSeg1 = numpy.concatenate((               MSeg1,                         numpy.zeros((len(sample["inp"]["seq1"]), 
                                                                                             len(sample["inp"]["seq2"]),
                                                                                             len(AminsCode)*2+1), dtype = numpy.float32)), axis=1)
        MSeg2 = numpy.concatenate((numpy.zeros((len(sample["inp"]["seq2"]), 
                                                len(sample["inp"]["seq1"]), 
                                                len(AminsCode)*2+1), dtype = numpy.float32),        MSeg2), axis=1)

        OSeg1 = numpy.concatenate((numpy.zeros((len(sample["inp"]["seq1"]), len(sample["inp"]["seq1"]))), sample["out"]), axis=1)
        OSeg2 = numpy.concatenate((numpy.transpose(sample["out"]), numpy.zeros((len(sample["inp"]["seq2"]), len(sample["inp"]["seq2"])))), axis=1)
        
        #print(OSeg1.shape, OSeg2.shape, sample["inp"]["in_dis1"].shape)
        ret = {
                "head": sample["head"],
                "inp": numpy.concatenate((MSeg1, MSeg2), axis = 0),
                "out": numpy.concatenate((OSeg1, OSeg2), axis = 0)
            }
        #logging.info("Data echo " + str(sample["head"]) + str(ret["inp"].shape))
        return ret
        
    def __len__(self):
        if self.test_mode:
            return len(self.data)
        else:
            return self.num_iters

def test_collate_fn_padd(Datas, check_inds: list):
    curbatch = []
    for i in check_inds:
        curbatch.append([Datas[i]["inp"], Datas[i]["out"]])
        print(curbatch[-1][0].shape, curbatch[-1][1].shape)
    max_len = max([hdim[0].shape[0] for hdim in curbatch])
    tail_shape = curbatch[0][0].shape[2:]
    for i in range(len(curbatch)):
        cur_len = curbatch[i][0].shape[0]
        #print(cur_len, max_len, (max_len-cur_len, cur_len)+tail_shape)
        #print(torch.full(((max_len-cur_len, cur_len)+tail_shape), 0).shape)
        u = torch.cat((torch.cat((torch.Tensor(curbatch[i][0]), torch.full(((max_len-cur_len, cur_len)+tail_shape), 0))), torch.full(((max_len, max_len-cur_len)+tail_shape), 0)), dim=1)
        o = torch.cat((torch.cat((torch.Tensor(curbatch[i][1]), torch.full(((max_len-cur_len, cur_len)),            0))), torch.full(((max_len, max_len-cur_len))           , 0)), dim=1)
        print("!!!", Datas[i]["out"].shape, o.shape)
        plt.pcolormesh(o, cmap='inferno')
        plt.show()       
    sys.exit()
        
if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
    logging.basicConfig(filename= str(Path(__file__).with_suffix(".log").name), filemode='w', level = logging.INFO, force=True,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.captureWarnings(capture=True)    
    testData = ProtContactDataSet("tempcsv", test_mode = True)
    t_all = t = time.time()
    # test collate
    for bi in range(len(testData)//20):
        test_collate_fn_padd(testData, [i for i in range(bi*20, bi*20+20)])
    logging.info(time.time()-t_all)
    
    # test __getitem__
    sample = testData[0]
    plt.pcolormesh(sample["out"], cmap='inferno')
    plt.show()