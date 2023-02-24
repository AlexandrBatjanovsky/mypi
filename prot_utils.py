# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from Bio.PDB import MMCIFParser
import scipy.spatial.distance as sp_dist
import numpy

from collections import Counter
from joblib import dump
#import pickle
import pandas

#import warnings
import sys
import logging
from tqdm import tqdm
from pathlib import Path
import gzip
#import time

AminsCode = [
                ("A", 	"ALA", 	"Alanine"),
                ("B", 	"ASX", 	"Aspartic acid or Asparagine"),
                ("C", 	"CYS", 	"Cysteine"),
                ("D", 	"ASP", 	"Aspartic acid"),
                ("E", 	"GLU", 	"Glutamic Acid"),
                ("F", 	"PHE", 	"Phenylalanine"),
                ("G", 	"GLY", 	"Glycine"),
                ("H", 	"HIS", 	"Histidine"),
                ("I", 	"ILE", 	"Isoleucine"),
                ("K", 	"LYS", 	"Lysine"),
                ("L", 	"LEU", 	"Leucine"),
                ("M", 	"MET", 	"Methionine"),
                ("N", 	"ASN", 	"Asparagine"),
                ("P", 	"PRO", 	"Proline"),
                ("Q", 	"GLN", 	"Glutamine"),
                ("R", 	"ARG", 	"Arginine"),
                ("S", 	"SER", 	"Serine"),
                ("T", 	"THR", 	"Threonine"),
                ("V", 	"VAL", 	"Valine"),
                ("W", 	"TRP", 	"Tryptophan"),
                ("X", 	"XAA", 	"Any amino acid"),
                ("Y", 	"TYR", 	"Tyrosine"),
                ("Z", 	"GLX", 	"Glutamine or Glutamic acid")
]
AminsCodeThreeToOne = {_[1]:_[0] for _ in AminsCode}
AminsCodeOneToThree = {_[0]:_[1] for _ in AminsCode}
AminsCodeOneToId = {_[0]:_i for _i, _ in enumerate(AminsCode)}
AminsCodeThreeToId = {_[1]:_i for _i, _ in enumerate(AminsCode)}

# create amino sequence layers
def Amins_Seq_to_Code_Line(Seq: list):
    AminsCounter = Counter(Seq)
    if not (set(AminsCounter.keys()) <= set(AminsCodeThreeToOne.keys())):
        for amini, amin in enumerate(Seq):
            if amin not in AminsCodeThreeToOne.keys(): Seq[amini] = "XAA"
        logging.warning("Error Unknown Amins" + str(set(AminsCounter.keys()) - set(AminsCodeThreeToOne.keys())))
    CodeSeq = [AminsCodeThreeToId[_] for _ in Seq]

    return CodeSeq, AminsCounter

# read protein models coords, seqs
# evaluate internal distances and PPI
def Process_Data_from_PDB(file_path_name: Path, PDB_parser = None, takeallmodels = False):
    # print(file_path_name)
    if PDB_parser is None:
        PDB_parser = MMCIFParser(QUIET=False)
    # read PDBFile if not flag(takeallmodels) load one first model
    with gzip.open(file_path_name, "rt") as gzcif_f:
        prot_BioData = list(PDB_parser.get_structure(file_path_name.name, gzcif_f).get_models())
    if len(prot_BioData) > 1:
        if not takeallmodels:
            prot_BioData = [prot_BioData[0]]
        else:
            logging.info("MultiModels: " + str(len(prot_BioData)))
    
    RetDate = []
    AminsCounter = Counter()
    # protein models run
    for modeli in prot_BioData:
        # extract all chains Ids for check uniqueness
        chainIDs = [_.get_id() for _ in modeli.get_chains()]
        if len(chainIDs) != len(set(chainIDs)):
            logging.warning(str(file_path_name) + " Some ChainIds")
            continue
        
        # Coords all atoms and CB-atoms (CA for Gly)
        # Sequence and seq datalayers
        # test breaks in chains
        chain_ca_atoms = {}
        
        chain_all_atoms_coord = {}
        atoms_res_divide_lens = {}
        
        chain_seq = {}
        seq_code_line = {}
        
        for chaini in modeli.get_chains():

            #tempCa = [x for x in chaini.get_atoms() 
            #              if ((x.get_parent().resname == 'GLY' and x.name == 'CA') or 
            #                  (x.name == 'CB')) and (not x.get_full_id()[3][0].strip())]
            
            tempCa = [x for x in chaini.get_atoms() 
                          if (x.name == 'CA') and (not x.get_full_id()[3][0].strip())]
            
            if not tempCa:
                logging.info(chaini.get_id() + "absence CA-atoms")
                continue
            chain_ca_atoms[chaini.get_id()] = tempCa

            # Sequence
            chain_seq[chaini.get_id()] = [atom_ca.get_parent().get_resname() for atom_ca in chain_ca_atoms[chaini.get_id()]]
            # DataLayer Sequence, index unknown amins, counter amins
            seq_code_line[chaini.get_id()], acurAminsCounter = Amins_Seq_to_Code_Line(chain_seq[chaini.get_id()])
            AminsCounter = AminsCounter + acurAminsCounter
            # delete unknown amins. change: replace with arbitrary
            # for amini in reversed(aminiExclude):
            #    del chain_ca_atoms[chaini.get_id()][amini]
            
            # all atoms(without H) model
            chain_all_atoms_coord[chaini.get_id()] = []
            for ca_atom in chain_ca_atoms[chaini.get_id()]:
                chain_all_atoms_coord[chaini.get_id()].extend([_.coord for _ in ca_atom.get_parent().get_atoms() if _.name != "H"])
                        
            # division by residues to calculate residue distances over atoms
            atoms_res_divide_lens[chaini.get_id()] = \
                [len([__ for __ in _.get_parent().get_atoms() if __.name != "H"]) for _ in chain_ca_atoms[chaini.get_id()]]
            #debug sum divide res/atoms and length all atom
            #if (sum(atoms_res_divide_lens[chaini.get_id()]) != len(chain_all_atoms_coord[chaini.get_id()])):
            #    print(sum(atoms_res_divide_lens[chaini.get_id()]), len(chain_all_atoms_coord[chaini.get_id()]))
            #    print("Debug!!!Error atoms!!!!")
            #    sys.exit()
            
            #chain break
            resi_in_chain = [atom_ca.get_parent().get_full_id()[3][1] for atom_ca in chain_ca_atoms[chaini.get_id()]]
            for resi in range(len(resi_in_chain)-1):
                if resi_in_chain[resi+1] - resi_in_chain[resi] != 1:
                    logging.warning(str(file_path_name) + " break in chain" + chaini.get_id() 
                                    + " resi " + str((resi_in_chain[resi], resi_in_chain[resi+1])))
                
        # internal distances and PPI calculate
        Intern_Dist = {}
        PPI = {}
        for chainID in chain_ca_atoms.keys():
            for checked_chains in Intern_Dist.keys():
                mutual_dists = sp_dist.cdist(chain_all_atoms_coord[chainID], chain_all_atoms_coord[checked_chains], 'euclidean')
                Interface = numpy.zeros((len(chain_ca_atoms[chainID]), len(chain_ca_atoms[checked_chains])))
                f_i = 0
                for fa, f_atomres_divide in enumerate(atoms_res_divide_lens[chainID]):
                    s_i = 0
                    for sa, s_atomres_divide in enumerate(atoms_res_divide_lens[checked_chains]):
                        res_res_dist = mutual_dists[f_i:f_i+f_atomres_divide, s_i:s_i+s_atomres_divide]
                        if numpy.any(res_res_dist < 4):
                            Interface[fa, sa] = 1
                        #else:
                        #    Interface[fa, sa] = 0
                        s_i = s_i + s_atomres_divide
                    f_i = f_i + f_atomres_divide
                PPI[(chainID, checked_chains)] = Interface
            ca_coord = numpy.array([catom.coord for catom in chain_ca_atoms[chainID]])
            Intern_Dist[chainID] = sp_dist.cdist(ca_coord, ca_coord, 'euclidean')
        
        #RetDate.append({"SLayers": seq_data_layer, "InDists": Intern_Dist, "PPIs": PPI})
        RetDate.append({"Seqs": seq_code_line, "InDists": Intern_Dist, "PPIs": PPI})
    
    
    #print([AminsCode[_i] for _i, _ in enumerate(RetDate[0]["SLayers"]["B"][2][3][0:23]) if _ == 1])  #[:, ...][0:23]  указывает на последовательность
    #print([AminsCode[_i] for _i, _ in enumerate(RetDate[0]["SLayers"]["B"][3][0][23:46]) if _ == 1]) #[..., :][23:46] указывает на последовательность
    
    return RetDate, AminsCounter

def dumpEvalDate(gzpath: Path, ProtDataDump: list, save_table_file = None):
    #print(gzpath)
    path = gzpath.parents[0]
    #print(path)
    dfilename = Path(gzpath.stem).stem
    with open(Path(str(path)+"/"+dfilename+".jbl"), "wb") as jblFile:
        dump(ProtDataDump, jblFile, compress = 3)
    if save_table_file:
        ProtTables = []
        for modeli in range(len(ProtDataDump)):
            for iteractchains in ProtDataDump[modeli]["PPIs"]:
                ProtTables.append([dfilename, str(path)+"/"+dfilename+".jbl", len(ProtDataDump), modeli, iteractchains[0], iteractchains[1]])
        pandas.DataFrame(ProtTables, columns = ["struct", "filename", "models", "imodel", "chain1", "chain2"]).to_csv(save_table_file, mode='a', index=False, header=False)
        
if __name__ == '__main__':
    logging.basicConfig(filename= str(Path(__file__).with_suffix(".log").name), filemode='w', level = logging.INFO, force=True,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.captureWarnings(capture=True)
    
    AminsCounter = Counter()
    for gzpath in tqdm(list(Path("/home/alexandersn/WORK/Python/ProtIter/mypi/ASSPFullConv/asm/").glob("*.gz"))):
    #for gzpath in list(Path("/home/alexandersn/WORK/Python/ProtIter/mypi/ASSPFullConv/").glob("*.gz")):   
    #for gzpath in tqdm(list(Path("/home/alexandersn/WORK/DATA/testGa100").glob("*.cif"))):
        # if cif file exist - read him
        logging.info("Check: " + str(gzpath.name))
        curProtData, curAminsCounter = Process_Data_from_PDB(gzpath, takeallmodels = True)
        AminsCounter = AminsCounter + curAminsCounter
        dumpEvalDate(gzpath, curProtData, "tempcsv")
    logging.info("Amin Content:" + str(AminsCounter))
    logging.info("Amin Table absence amins:" + str( set([_[1] for _ in AminsCode]).difference(set(AminsCounter.keys())) ))