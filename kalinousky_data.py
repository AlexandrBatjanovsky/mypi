#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:11:07 2023

@author: alexandersn
"""

import pandas
import numpy
from torch.utils.data import Dataset
#import logging
#import time
from joblib import load

from prot_utils import AminsCode

#import sys

class ProtContactDataSet(Dataset):
    def __init__(self, dat_files_csv: str, test_mode = False , num_iters=100):
        self.dat_files_csv = dat_files_csv
        self.test_mode = test_mode
        self.num_iters = num_iters
        
        # load csv with pair globuls
        self.CSVFileData = pandas.read_csv(self.dat_files_csv)
        self.CSVFileData.columns = ["struct", "filename", "models", "imodel", "chain1", "chain2"]
        self.data = []
        
        # load data of pair globuls
        for c_rec in self.CSVFileData.groupby("filename"):
            jbl_dat = load(c_rec[0])
            for index, row in c_rec[1].iterrows():
                self.data.append({"inp": {"seq1": jbl_dat[row["imodel"]]["Seqs"][row["chain1"]], 
                                          "seq2": jbl_dat[row["imodel"]]["Seqs"][row["chain2"]],
                                          "in_dis1": jbl_dat[row["imodel"]]["InDists"][row["chain1"]],
                                          "in_dis2": jbl_dat[row["imodel"]]["InDists"][row["chain2"]]},
                                  "out": jbl_dat[row["imodel"]]["PPIs"][(row["chain1"], row["chain2"])]})
                
    def __getitem__(self, item):
        
        # create sequence code layers (2*Aminotable layers)
        def AminsCodeLine_to_AminsCodeLayers(SeqCode):
            HorSeq, VerSeq = numpy.meshgrid(SeqCode, SeqCode)
            shp_inp = HorSeq.shape
            HorSeq = numpy.eye(len(AminsCode))[HorSeq.reshape(-1)]
            VerSeq = numpy.eye(len(AminsCode))[VerSeq.reshape(-1)]
            HorSeq = HorSeq.reshape(shp_inp + (len(AminsCode),))
            VerSeq = VerSeq.reshape(shp_inp + (len(AminsCode),))
            TensorSeq = numpy.dstack([HorSeq, VerSeq])
            return TensorSeq
        
        sample = self.data[item]
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
        ret = {"inp": numpy.concatenate((MSeg1, MSeg2), axis = 0),
               "out": sample["out"]}
        return ret
        
    def __len__(self):
        if self.test_mode:
            return len(self.data)
        else:
            return self.num_iters

if __name__ == '__main__':
    testData = ProtContactDataSet("tempcsv", test_mode = True)
    for eve in range(len(testData)):
        testData[eve]