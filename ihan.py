import time
from datasets import load_dataset,DatasetDict, load_metric
#import evaluate
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch
import torch.nn as nn
from torchmetrics.functional import auroc

import pandas as pd
import numpy as np
import pickle
import json

from transformers import AdamW,get_scheduler, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from accelerate import Accelerator, DistributedType, notebook_launcher
import random
import math
import statistics
import copy
from functools import partial
import csv
from pathlib import Path

#target="hcc138"

#Leave the cd1,...,cd4 order as it is. Only some are used when not using all 4 code types
#cd1, cd3, cd4 are used in 3 codes
#cd1 = 'diag'
#num_codes1 = 42683 + 1 
#cd2 = 'proc'
#num_codes2 = 29231+ 1
#cd3 = 'lab'
#num_codes3 = 12263 + 1
#cd4 = 'rx'
#num_codes4 = 1879 + 1

#nvisit_min = 2
#nvisit_max = 365*3
#max_ncodes_perVisit = 512
#r_0to1 = 3        # ratio of #target=0 to #target=1 in data
#r_val = 0.20      # fraction of data in  validation
#r_test = 0.20     # fraction of data in testing
#r_train = 1 - r_val - r_test   # fraction of data in training


######Data prep functions
def seriesToList(x):
    """
    set to empty list if element of a series is not a list
    """
    idx = x.apply(type) == list
    xx=pd.Series("[]", index = range(len(x)))
    xx = xx.apply(eval)
    xx[idx] = x[idx]
    xx = xx.values.tolist()
    return xx

def xList_yList(dfPandas, id = 'mcid', tgt = 'label', code_idxs=['datList1','datList2','datList3','datList4']):
    """
    This code creates from a dataframe a list for each column specified in id, tgt, and code_idxs.
    xlist: list(list(list)), list of patients, list of visits, list of notes/codes, ... 
    ylist = list(labels)
    """
    idList = dfPandas[id].tolist()
    targetList = dfPandas[tgt].tolist()
    #dfPandas[code_idxs[0]]= dfPandas[code_idxs[0]].apply(ast.literal_eval)
    dataList1 = seriesToList(dfPandas[code_idxs[0]])
    if code_idxs[1:] == [None, None, None]:
        return idList, targetList, dataList1
    elif code_idxs[2:] == [None, None]:
        dataList2 = seriesToList(dfPandas[code_idxs[1]])
        return idList, targetList, dataList1, dataList2 
    elif code_idxs[3:] == [None]:
        dataList2 = seriesToList(dfPandas[code_idxs[1]])
        dataList3 = seriesToList(dfPandas[code_idxs[2]])
        return idList, targetList, dataList1, dataList2, dataList3
    else:
        dataList2 = seriesToList(dfPandas[code_idxs[1]])
        dataList3 = seriesToList(dfPandas[code_idxs[2]])
        dataList4 = seriesToList(dfPandas[code_idxs[3]])
        return idList, targetList, dataList1, dataList2, dataList3, dataList4
    
def xList_yList_json(dfPandas, id = 'mcid', tgt = 'label', code_idxs=['datList1','datList2','datList3','datList4']):
    """This code create 
    xlist: list(list(list)), list of patients, list of visits, list of notes/codes, ... 
    ylist = list(labels)
    """
    import json
    idList = dfPandas[id].tolist()
    targetList = dfPandas[tgt].tolist()
    dataList1 = dfPandas[code_idxs[0]].apply(json.loads).values.tolist()
    if code_idxs[1:] == [None, None, None]:
        return idList, targetList, dataList1
    elif code_idxs[2:] == [None, None]:
        dataList2 = dfPandas[code_idxs[1]].apply(json.loads).values.tolist()
        return idList, targetList, dataList1, dataList2 
    elif code_idxs[3:] == [None]:
        dataList2 = dfPandas[code_idxs[1]].apply(json.loads).values.tolist()
        dataList3 = dfPandas[code_idxs[2]].apply(json.loads).values.tolist()
        return idList, targetList, dataList1, dataList2, dataList3
    else:
        dataList2 = dfPandas[code_idxs[1]].apply(json.loads).values.tolist()
        dataList3 = dfPandas[code_idxs[2]].apply(json.loads).values.tolist()
        dataList4 = dfPandas[code_idxs[3]].apply(json.loads).values.tolist()
        return idList, targetList, dataList1, dataList2, dataList3, dataList4
    
def xList_yList2(dfPandas, id = 'mcid', tgt = 'label', code_idxs=['datList1','datList2','datList3','datList4'],
                doss = ['dos1', 'dos2', 'dos3', 'dos4']):
    """This code create 
    xlist: list(list(list)), list of patients, list of visits, list of notes/codes, ... 
    ylist = list(labels)
    for interpretation use: also with dos list in addition to output from xlist_ylist
    """
    idList = dfPandas[id].tolist()
    targetList = dfPandas[tgt].tolist()
    dosList1 = seriesToList(dfPandas[doss[0]])
    dataList1 = seriesToList(dfPandas[code_idxs[0]])
    if code_idxs[1:] == [None, None, None]:
        return idList, targetList, dosList1, dataList1
    elif code_idxs[2:] == [None, None]:
        dosList2 = seriesToList(dfPandas[doss[1]])
        dataList2 = seriesToList(dfPandas[code_idxs[1]])
        return idList, targetList, dosList1, dataList1, dosList2, dataList2 
    elif code_idxs[3:] == [None]:
        dosList2 = seriesToList(dfPandas[doss[1]])
        dataList2 = seriesToList(dfPandas[code_idxs[1]])
        dosList3 = seriesToList(dfPandas[doss[2]])
        dataList3 = seriesToList(dfPandas[code_idxs[2]])
        return idList, targetList, dosList1, dataList1, dosList2, dataList2, dosList3, dataList3
    else:
        dosList2 = seriesToList(dfPandas[doss[1]])
        dataList2 = seriesToList(dfPandas[code_idxs[1]])
        dosList3 = seriesToList(dfPandas[doss[2]])
        dataList3 = seriesToList(dfPandas[code_idxs[2]])
        dosList4 = seriesToList(dfPandas[doss[3]])
        dataList4 = seriesToList(dfPandas[code_idxs[3]])
        return idList, targetList, dosList1, dataList1, dosList2, dataList2, dosList3, dataList3, dosList4, dataList4
    
    
#Use the dataList01 as driver
#pick members only if number of visits >= nvisit_min & <= nvisit_max based on dataList01
def selMembers(label0, dataList01, dataList02, dataList03, dataList04, nvisit_min=2, nvisit_max=365*3):
    label = [y for x1, y in zip(dataList01, label0) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]   
    dataList1 = [x1 for x1 in dataList01 if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    dataList2 = [x2 for x1, x2 in zip(dataList01, dataList02) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    dataList3 = [x3 for x1, x3 in zip(dataList01, dataList03) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    dataList4 = [x4 for x1, x4 in zip(dataList01, dataList04) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    return label, dataList1, dataList2, dataList3, dataList4

def selMembers2(idList0, label0, dataList01, dataList02, dataList03, dataList04, nvisit_min=2, nvisit_max=365*3):
    """ add idList0 from  selMembers"""
    label = [y for x1, y in zip(dataList01, label0) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    idList = [id for x1, id in zip(dataList01, idList0) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    dataList1 = [x1 for x1 in dataList01 if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    dataList2 = [x2 for x1, x2 in zip(dataList01, dataList02) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    dataList3 = [x3 for x1, x3 in zip(dataList01, dataList03) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    dataList4 = [x4 for x1, x4 in zip(dataList01, dataList04) if (len(x1) >= nvisit_min) & (len(x1) <= nvisit_max)]
    return idList, label, dataList1, dataList2, dataList3, dataList4


#balance data 
def balance_label(label, dtList1, dtList2, dtList3, dtList4, r=3):
    """
    Balance the data such that the ratio of label=1 to label=0 is 1 to r
    """
    
    #take all label = 1
    label1 = [y for y in label if y == 1]
    dtList1_1 = [x for x, y in zip(dtList1, label) if y == 1]
    dtList2_1 = [x for x, y in zip(dtList2, label) if y == 1]
    dtList3_1 = [x for x, y in zip(dtList3, label) if y == 1]
    dtList4_1 = [x for x, y in zip(dtList4, label) if y == 1]
    n1 = len(label1)
    print("Count of label = 1:", n1)

    #take r*n1 of label = 0
    n0 = r*n1
    label0 = [y for y in label if y == 0]
    dtList1_0 = [x for x, y in zip(dtList1, label) if y == 0]
    dtList2_0 = [x for x, y in zip(dtList2, label) if y == 0]
    dtList3_0 = [x for x, y in zip(dtList3, label) if y == 0]
    dtList4_0 = [x for x, y in zip(dtList4, label) if y == 0]
    c = list(zip(dtList1_0, dtList2_0, dtList3_0,dtList4_0,label0))
    random.shuffle(c)
    dtList1_0_shuffled, dtList2_0_shuffled, dtList3_0_shuffled,dtList4_0_shuffled,label0_shuffled = zip(*c)
    
    #append *1 and *0_shuffled
    dtList1_1.extend(dtList1_0_shuffled[:n0])
    dtList2_1.extend(dtList2_0_shuffled[:n0])
    dtList3_1.extend(dtList3_0_shuffled[:n0])
    dtList4_1.extend(dtList4_0_shuffled[:n0])
    label1.extend(label0_shuffled[:n0])

    #randomize the order
    c = list(zip(dtList1_1, dtList2_1, dtList3_1, dtList4_1,label1))
    random.shuffle(c)
    datList1, datList2, datList3, datList4, label = zip(*c)
    datList1 =list(datList1)
    datList2 =list(datList2)
    datList3 =list(datList3)
    datList4 =list(datList4)
    label = list(label)
    
    return label, datList1, datList2, datList3, datList4


def balance_label2(idList, label, dtList1, dtList2, dtList3, dtList4, r=3):
    """
    Balance the data such that the ratio of label=1 to label=0 is 1 to r
    add idList from balance_label
    """
    
    #take all label = 1
    label1 = [y for y in label if y == 1]
    idList_1 =  [x for x, y in zip(idList, label) if y == 1]
    dtList1_1 = [x for x, y in zip(dtList1, label) if y == 1]
    dtList2_1 = [x for x, y in zip(dtList2, label) if y == 1]
    dtList3_1 = [x for x, y in zip(dtList3, label) if y == 1]
    dtList4_1 = [x for x, y in zip(dtList4, label) if y == 1]
    n1 = len(label1)
    print("Count of label = 1:", n1)

    #take r*n1 of label = 0
    n0 = r*n1
    label0 = [y for y in label if y == 0]
    idList_0 =  [x for x, y in zip(idList, label) if y == 0]
    dtList1_0 = [x for x, y in zip(dtList1, label) if y == 0]
    dtList2_0 = [x for x, y in zip(dtList2, label) if y == 0]
    dtList3_0 = [x for x, y in zip(dtList3, label) if y == 0]
    dtList4_0 = [x for x, y in zip(dtList4, label) if y == 0]
    c = list(zip(dtList1_0, dtList2_0, dtList3_0,dtList4_0,label0,idList_0))
    random.shuffle(c)
    dtList1_0_shuffled, dtList2_0_shuffled, dtList3_0_shuffled,dtList4_0_shuffled,label0_shuffled,idList_0_shuffled = zip(*c)
    
    #append *1 and *0_shuffled
    idList_1.extend(idList_0_shuffled[:n0])
    dtList1_1.extend(dtList1_0_shuffled[:n0])
    dtList2_1.extend(dtList2_0_shuffled[:n0])
    dtList3_1.extend(dtList3_0_shuffled[:n0])
    dtList4_1.extend(dtList4_0_shuffled[:n0])
    label1.extend(label0_shuffled[:n0])

    #randomize the order
    c = list(zip(dtList1_1, dtList2_1, dtList3_1, dtList4_1,label1,idList_1))
    random.shuffle(c)
    datList1, datList2, datList3, datList4, label, idList = zip(*c)
    datList1 =list(datList1)
    datList2 =list(datList2)
    datList3 =list(datList3)
    datList4 =list(datList4)
    idList = list(idList)
    label = list(label)
    
    return idList,label, datList1, datList2, datList3, datList4

#Summary
def code_summary(dataList):
    num_patients = len(dataList)
    #all_codes = list(set([j for i in sum(codeList, []) for j in i]))
    num_visits = [len(patient) for patient in dataList]
    num_codes = [len(visit) for patient in dataList for visit in patient]   #per visit
    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)
    min_num_visits = min(num_visits)
    min_num_codes = min(num_codes)
    md_num_visits = statistics.median(num_visits)
    md_num_codes = statistics.median(num_codes)
    #max and min index in dataList
    max_idx_visit = [max(visit) for patient in dataList for visit in patient]
    max_idx = max(max_idx_visit)
    min_idx_visit = [min(visit) for patient in dataList for visit in patient]
    min_idx = min(min_idx_visit)

    return num_patients, md_num_visits, min_num_visits, max_num_visits, md_num_codes, min_num_codes,max_num_codes, min_idx, max_idx

######Custom data and data collate function   
class CustomDataset(Dataset):
    def __init__(self, data, tgt = 'label', 
                 cols_seqCode = ['datList1','datList2','datList3','datList4'],
                 cols_seqPair = ['labPairList'],
                 cols_statFeature = ['num_age','num_genderm']):
        """
        data: data frame
        tgt: column name of target variable. Maybe None
        cols_seqCode: column names for sequenctial medical codes: diagnosis sequence, procedure, lab, medication. At least one should occur.
        cols_seqPair: column names for sequenctial pair data such as lab: loinc_cd: test_value. Maybe None
        cols_statFeature: column names for static features such as age, gender. Maybe None
        
        output: (seq1, ..., seq_n, label) or (seq1, ..., seq_n, seqPair_key1, seqPair_value1, ..., staticX, label)
        """
        self.tgt = tgt
        self.cols_statFeature = cols_statFeature
        
        self.num_seqs = 0 if cols_seqCode is None else len(cols_seqCode)
        self.num_seqPairs = 0 if cols_seqPair is None else len(cols_seqPair)     
        self.num_statFeature = 0 if cols_statFeature is None else len(cols_statFeature)
        
        if tgt is not None:
            self.label = data[tgt].tolist()
        if self.num_seqs > 0:
            for i, col in enumerate(cols_seqCode):
                setattr(self, f"seq{i+1}", data[col].apply(json.loads).values.tolist())
        if self.num_seqPairs > 0:
            for i, col in enumerate(cols_seqPair):
                seqPairList = data[col].apply(eval).values.tolist()
                k = [[[x for x, y in v.items()] for v in m] for m in seqPairList]
                v = [[[y for x, y in v.items()] for v in m] for m in seqPairList]
                setattr(self, f"seqPair_key{i+1}", k)
                setattr(self, f"seqPair_value{i+1}", v)
        if self.num_statFeature > 0:
            #self.staticX = np.array(data[cols_statFeature])
            self.staticX = np.array(data[cols_statFeature].values)
            #self.num_statFeature = len(cols_statFeature)
            
    def __len__(self):
        #return  len(self.seq1)
        return len(self.seq1) if self.num_seqs > 0 else len(self.seqPair_key1)

    def __getitem__(self, index):
        output = ()
        for i in range(1, self.num_seqs + 1):
            output += (getattr(self, f"seq{i}")[index],)
        for i in range(1, self.num_seqPairs + 1):
            output += (getattr(self, f"seqPair_key{i}")[index],)    
            output += (getattr(self, f"seqPair_value{i}")[index],) 
        if self.num_statFeature > 0:
            output += (self.staticX[index],)
        if self.tgt is not None:
            output += (self.label[index],)
        #for x in output:
        #    print(type(x))
        return output
    
    def __str__(self):
        return "length: {}, Num of sequential codes: {}, nume of sequential pair: {}, num of static features: {}".format(len(self), self.num_seqs, self.num_seqPairs, self.num_statFeature)   

    
def prepare_seq_x_and_mask(sequence, max_ncodes_per_visit=512):
    """
    For one sequnce of code integer index, create the tensor x and its corrsponding mask tensor
    sequence: list(list(list())), list of patients, list of visits per patient, list of codes per patient and visit
    """
    num_patients = len(sequence)
    num_visits = [len(patient) for patient in sequence]
    num_codes = [len(visit) for patient in sequence for visit in patient ]
    max_num_visits = max(max(num_visits), 1)
    max_num_codes = min(max(num_codes), max_ncodes_per_visit) if len(num_codes) > 0 else 1
    #print(num_patients, max_num_visits, max_num_codes)
        
    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    
    for i_patient, patient in enumerate(sequence):
        for j_visit, visit in enumerate(patient):
            if len(visit) > max_num_codes:
                visit = visit[:max_num_codes]
            x[i_patient, j_visit, :len(visit)] = torch.tensor(visit)
            masks[i_patient, j_visit, :len(visit)] = True  
    return x, masks

def prepare_seqPair_x_and_mask(sequence_key, sequence_value, max_ncodes_per_visit=512):
    """
    For one sequnce of float value, create the tensor x
    Arguments:
        sequence_key: list(list(list())), list of patients, list of visits per patient, list of codes per patient and visit
        sequence_value: list(list(list())), same structure as sequence_key, but it's list of values correspoding to the list of codes.
    outputs:
        x, masks, x_value: all with shape = (num_patients, max_num_visits, max_num_codes)
    """
    num_patients = len(sequence_key)
    num_visits = [len(patient) for patient in sequence_key]
    num_codes = [len(visit) for patient in sequence_key for visit in patient ]
    max_num_visits = max(max(num_visits), 1)
    max_num_codes = min(max(num_codes), max_ncodes_per_visit) if len(num_codes) > 0 else 1
    #print(num_patients, max_num_visits, max_num_codes)
        
    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    x_value = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.float)
    
    for i_patient, patient in enumerate(sequence_key):
        for j_visit, visit in enumerate(patient):
            if len(visit) > max_num_codes:
                visit = visit[:max_num_codes]
            x[i_patient, j_visit, :len(visit)] = torch.tensor(visit)
            masks[i_patient, j_visit, :len(visit)] = True 
    for i_patient, patient in enumerate(sequence_value):
        for j_visit, visit in enumerate(patient):
            if len(visit) > max_num_codes:
                visit = visit[:max_num_codes]
            x_value[i_patient, j_visit, :len(visit)] = torch.tensor(visit)
    return x, masks, x_value
   

def collate_fn(data, num_seq_vars=3, num_seqPairs_vars=1, statFeature=True, tgt = True, max_ncodes_perVisit = 512):
    """ 
    Arguments:
        data: a list of samples fetched from `CustomDataset` 
        num_seq_vars: number of sequential codes,
        num_seqPairs_vars: number of sequential pairs,
        statFeature: True or False for with or without static features
        tgt: True or False for with or without the target in data
    Outputs:
        output: a tuple of tensors. 
            xs -- tensor of shape (# patiens, max # visits, max # codes)), 
            masks -- tensor of shape (# patiens, max # visits, max # codes) of type torch.bool, 
            x_static -- tensor of shape (# patiens, # features)
            y -- a tensor of shape (# patiens) of type torch.float
    """
    if tgt:
        if statFeature:
            *sequences, staticXs, labels = zip(*data)
            x_static = torch.tensor(np.array(staticXs), dtype=torch.float)
        else:
            *sequences, labels = zip(*data)
        y = torch.tensor(labels, dtype=torch.float)
    else:
        if statFeature:
            *sequences, staticXs = zip(*data)
            x_static = torch.tensor(np.array(staticXs), dtype=torch.float)
        else:
            sequences = zip(*data)
            
    output = []
    #for seqcode
    for seq in sequences[:num_seq_vars]:  
        x, mask = prepare_seq_x_and_mask(seq, max_ncodes_perVisit)  
        output.extend([x, mask])  
    #for seqPair
    for i in range(num_seqPairs_vars):
        seqPair_key = sequences[num_seq_vars + 2*i]
        seqPair_value = sequences[num_seq_vars + 2*i+1]
        x_key_p, mask_key_p, x_value_p = prepare_seqPair_x_and_mask(seqPair_key, seqPair_value, max_ncodes_perVisit)
        output.extend([x_key_p, mask_key_p, x_value_p])  
    if statFeature:
        output.append(x_static)
    if tgt:
        output.append(y)  
    return tuple(output)              
            

######Attention function and sum
class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        """
        Arguments:
            hidden_dim: the hidden dimension
        """
        self.att = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, masks):
        """
        Arguments:
            x: the input tensor of shape = (batch_size, num_visits, (num_notes), hidden_dim) for codes and visits within each type
               shape = (batch_size, hidden_dim) for type
            masks: shape = (batch_size, num_visits, (num_notes))
        Outputs: the corresponding attention weights of shape = (batch_size, num_visits, (num_notes), 1)
        """
        s = x.shape
        l = len(masks.shape)
        alpha = torch.zeros(list(s[:(-1)])+[1], dtype=torch.float).to(str(x.device))
        if l == 3:
            m = masks[:,0,0]   #for codes and visits within each type, indicator if there is any visit for a member
        else:
            m = torch.sum(masks, dim = -1) > 0     #for types
        xx = x[m]         #non-empty part of x
        if len(xx) > 0:
            alpha1 = torch.softmax(self.att(xx),dim= -2)
            alpha[m] = alpha1
        
        return alpha
    
def attention_sum(alpha, x, masks=None):
    """
    without mask
    mask select for true visits (not padding visits) and then sum them up.
    Arguments:
        alpha: the alpha attention weights of shape (batch_size, #visits, 1)
        x: the visit embeddings of shape (batch_size, #visits, embedding_dim)
        masks: the padding masks of shape (batch_size, #visits, #codes)
    Outputs:
        c: the context vector of shape (batch_size, hidden_dim)
    """
    c = torch.sum(alpha * x, dim = -2)
    return c

def attention_visits_sum(alpha, x, masks=None):
    """
    mask select for true visits (not padding visits) and then sum them up.
    Arguments:
        alpha: the alpha attention weights of shape (batch_size, #visits, 1)
        x: the visit embeddings of shape (batch_size, #visits, embedding_dim)
        masks: the padding masks of shape (batch_size, #visits, #codes)
    Outputs:
        c: the context vector of shape (batch_size, hidden_dim)
    """
    #masks1 = torch.broadcast_to(masks[:,:,0][...,None], alpha.shape)
    m1 = masks[:,:,0][:,:,None]
    c = torch.sum((alpha * x) * m1, dim = -2)
    #rescale attentions among non-zeros
    s = torch.sum(alpha * m1, dim = -2)
    #print(c.shape, s.shape)
    c = c/s
    c = c.nan_to_num(0.0)
    return c

def attention_visits_rescale(alpha, masks=None):
    """
    rescale the visit attentions among the true visits to sum to 1.
    mask select for true visits (not padding visits) and then sum them up.
    Arguments:
        alpha: the alpha attention weights of shape (batch_size, #visits, 1)
        masks: the padding masks of shape (batch_size, #visits, #codes)
    Outputs:
        alpha_rescaled:  (batch_size,  #visits, 1)
    """
    alpha_rescaled = torch.zeros(alpha.shape, dtype=torch.float).to(str(alpha.device))
    m1 = masks[:,:,0][:,:,None]            #(batch_size, #visits, 1)
    #rescale attentions among non-zeros
    s = torch.sum(alpha * m1, dim = -2)    #(batch_size, 1)
    #print(s.shape)
    tmp = alpha/s[:,:, None]
    alpha_rescaled[m1] = tmp[m1]
    return alpha_rescaled

def attention_codes_sum(alpha, x, masks=None):
    """
    mask select for true codes (not padding codes) and then sum them up.
    Arguments:
        alpha: the alpha attention weights of shape (batch_size, #visits, #codes, 1)
        x: the code embeddings of shape (batch_size, #visits, #codes, embedding_dim)
        masks: the padding masks of shape (batch_size, #visits, #codes)
    Outputs:
        c: the context vector of shape (batch_size, hidden_dim)
    """
    masks1 = masks[:,:,:,None]
    c = torch.sum((alpha * x) * masks1, dim = -2)
    #rescale attentions among non-zeros. rescale give cuda error when running, why?
    #s = torch.sum(alpha * masks1, dim = -2)
    #print(c.shape, s.shape)
    #c = c/s     
    #print(c.isnan().any())
    #c = c.nan_to_num(0.0)
    #print(c.isnan().any())
    return c


def attention_types_sum(alpha, x, masks=None):
    """
    mask select for true visits (not padding visits) and then sum them up.
    Arguments:
        alpha: the alpha attention weights of shape (batch_size, #types, 1)
        x: the visit embeddings of shape (batch_size, #types, embedding_dim)
        masks: the padding masks of shape (batch_size, #types)
    Outputs:
        c: the context vector of shape (batch_size, hidden_dim)
    """
    masks1 = masks[:,:,None]
    c = torch.sum((alpha * x) * masks1, dim = -2)
    #rescale attentions among non-zeros
    s = torch.sum(alpha * masks1, dim = -2)
    #print(c.shape, s.shape)
    c = c/s
    return c


######Custom model architecture
class CustomModel_1codes(nn.Module):
    """
    for binary classification
    input: 1 of multiple types of codes: diag (x1), proc (x2), lab (x3), rx (x4)
    """
    def __init__(self, num_codes1, embedding_dim=128): 
        super(CustomModel_1codes,self).__init__() 
        #self.num_labels = num_labels 
        self.embedding1 = nn.Embedding(num_codes1, embedding_dim, padding_idx=0)
        self.attn_note1 = Attention(embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attn_visit1 = Attention(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1) # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, masks1, interpret = False):
        """
        Arguments:
            x: the pre-trained model embeddings of notes sequence of shape (batch_size, # visits, # notes, embedding_dim)
            masks: the padding masks of shape (batch_size, # visits, # notes) 
        """
        #Add custom layers
        #print('')
        #print('start: CustomModel_2codes')
        #print('x1:',x1.device, '; masks1:',masks1.device, '; x2:',x2.device, '; masks2:',masks2.device)
        #print('')
        
        #print('x1',x1.shape, x1)
        xx1 = self.embedding1(x1)
        #print('xx1',xx1.shape, xx1.device)
        #print('')
        a_note1 = self.attn_note1(xx1, masks1)
        #print('a_note1',a_note1.shape,a_note1.device)
        #print('')
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        #print('')
        h1, _ = self.gru1(x1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        #print('a_visit1',a_visit1.shape,a_visit1.device)
        #print('')
        x = attention_visits_sum(a_visit1, x1, masks=masks1)
        #print('x1',x1.shape, x1.device)
                                   
        logits = self.linear(x) # shape = (batch_size, 1)
        ##print(logits.shape)
        prob = self.sigmoid(logits)
        prob = prob.squeeze(dim = -1)
        #print('prob:', prob.shape, prob)       

        if interpret:
            return prob, xx1, a_note1, a_visit1 
        else:
            return prob


class CustomModel_2codes(nn.Module):
    """
    for binary classification
    input: 2 of multiple types of codes: diag (x1), proc (x2), lab (x3), rx (x4)
    """
    def __init__(self, num_codes1, num_codes2, embedding_dim=128): 
        super(CustomModel_2codes,self).__init__() 
        #self.num_labels = num_labels 
        self.embedding1 = nn.Embedding(num_codes1, embedding_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(num_codes2, embedding_dim, padding_idx=0)
        self.attn_note1 = Attention(embedding_dim)
        self.attn_note2 = Attention(embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attn_visit1 = Attention(embedding_dim)
        self.attn_visit2 = Attention(embedding_dim)
        self.attn_type = Attention(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1) # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, masks1, x2, masks2, interpret = False):
        """
        Arguments:
            x: the pre-trained model embeddings of notes sequence of shape (batch_size, # visits, # notes, embedding_dim)
            masks: the padding masks of shape (batch_size, # visits, # notes) 
        """
        #Add custom layers
        #print('')
        #print('start: CustomModel_2codes')
        #print('x1:',x1.device, '; masks1:',masks1.device, '; x2:',x2.device, '; masks2:',masks2.device)
        #print('')
        
        #print('x1',x1.shape, x1)
        xx1 = self.embedding1(x1)
        #print('xx1',xx1.shape, xx1.device)
        #print('')
        a_note1 = self.attn_note1(xx1, masks1)
        #print('a_note1',a_note1.shape,a_note1.device)
        #print('')
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        #print('')
        h1, _ = self.gru1(x1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        #print('a_visit1',a_visit1.shape,a_visit1.device)
        #print('')
        x1 = attention_visits_sum(a_visit1, x1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        
        #print('x2',x2.shape, x2)
        xx2 = self.embedding2(x2)
        #print('xx2',xx2.shape, xx2)
        a_note2 = self.attn_note2(xx2, masks2)
        #print('a_note2',a_note2.shape,a_note2)
        x2 = attention_codes_sum(a_note2, xx2, masks=masks2)
        #print('x2',x2.shape, x2)
        h2, _ = self.gru2(x2)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit2 = self.attn_visit2(h2, masks2)                  #visit attention
        #print('a_visit2',a_visit2.shape,a_visit2)
        x2 = attention_visits_sum(a_visit2, x2, masks=masks2)
        #print('x2',x2.shape, x2.device)
        
        #stack different type of codes: diag, proc, lab, etc
        xx = torch.stack([x1, x2],dim = -2)
        #print('xx',xx.shape,xx)
        masks_type = torch.stack([masks1[:, 0, 0], masks2[:,0,0]], dim = 1)     #shape=(batch_size, #types)
        #print('masks_type',masks_type.shape, masks_type)
        #attention for different type
        a_type = self.attn_type(xx, masks_type)
        #print('a_type', a_type.shape, a_type)
        x = attention_types_sum(a_type, xx, masks=masks_type) 
        #print('x',x.shape, x.device)
                    
        logits = self.linear(x) # shape = (batch_size, 1)
        ##print(logits.shape)
        prob = self.sigmoid(logits)
        prob = prob.squeeze(dim = -1)
        #print('prob:', prob.shape, prob)       

        if interpret:
            return prob, xx1, a_note1, a_visit1, xx2, a_note2, a_visit2, a_type, masks_type 
        else:
            return prob


class CustomModel_3codes(nn.Module):
    """
    for binary classification
    input: 3 of multiple types of codes: diag (x1), proc (x2), lab (x3), rx (x4)
    """
    def __init__(self, num_codes1, num_codes2, num_codes3, embedding_dim=128): 
        super(CustomModel_3codes,self).__init__() 
        #self.num_labels = num_labels 
        self.embedding1 = nn.Embedding(num_codes1, embedding_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(num_codes2, embedding_dim, padding_idx=0)
        self.embedding3 = nn.Embedding(num_codes3, embedding_dim, padding_idx=0)        
        self.attn_note1 = Attention(embedding_dim)
        self.attn_note2 = Attention(embedding_dim)
        self.attn_note3 = Attention(embedding_dim)        
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru3 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)        
        self.attn_visit1 = Attention(embedding_dim)
        self.attn_visit2 = Attention(embedding_dim)
        self.attn_visit3 = Attention(embedding_dim)        
        self.attn_type = Attention(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1) # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, masks1, x2, masks2, x3, masks3, interpret = False):
        """
        Arguments:
            x: the pre-trained model embeddings of notes sequence of shape (batch_size, # visits, # notes, embedding_dim)
            masks: the padding masks of shape (batch_size, # visits, # notes) 
        """
        #Add custom layers
        #print('')
        #print('start: CustomModel_2codes')
        #print('x1:',x1.device, '; masks1:',masks1.device, '; x2:',x2.device, '; masks2:',masks2.device)
        #print('')
        
        #print('x1',x1.shape, x1)
        xx1 = self.embedding1(x1)
        #print('xx1',xx1.shape, xx1.device)
        #print('')
        a_note1 = self.attn_note1(xx1, masks1)
        #print('a_note1',a_note1.shape,a_note1.device)
        #print('')
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        #print('')
        h1, _ = self.gru1(x1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        #print('a_visit1',a_visit1.shape,a_visit1.device)
        #print('')
        x1 = attention_visits_sum(a_visit1, x1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        
        #print('x2',x2.shape, x2)
        xx2 = self.embedding2(x2)
        #print('xx2',xx2.shape, xx2)
        a_note2 = self.attn_note2(xx2, masks2)
        #print('a_note2',a_note2.shape,a_note2)
        x2 = attention_codes_sum(a_note2, xx2, masks=masks2)
        #print('x2',x2.shape, x2)
        h2, _ = self.gru2(x2)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit2 = self.attn_visit2(h2, masks2)                  #visit attention
        #print('a_visit2',a_visit2.shape,a_visit2)
        x2 = attention_visits_sum(a_visit2, x2, masks=masks2)
        #print('x2',x2.shape, x2.device)
        
        #print('x3',x3.shape, x3)
        xx3 = self.embedding3(x3)
        #print('xx3',xx3.shape, xx3.device)
        #print('')
        a_note3 = self.attn_note3(xx3, masks3)
        #print('a_note3',a_note3.shape,a_note3.device)
        #print('')
        x3 = attention_codes_sum(a_note3, xx3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        #print('')
        h3, _ = self.gru3(x3)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit3 = self.attn_visit3(h3, masks3)                  #visit attention
        #print('a_visit3',a_visit3.shape,a_visit3.device)
        #print('')
        x3 = attention_visits_sum(a_visit3, x3, masks=masks3)
        #print('x3',x3.shape, x3.device)
               
        #stack different type of codes: diag, proc, lab, etc
        xx = torch.stack([x1, x2, x3],dim = -2)
        #print('xx',xx.shape,xx)
        masks_type = torch.stack([masks1[:, 0, 0], masks2[:,0,0], masks3[:,0,0]], dim = 1)     #shape=(batch_size, #types)
        #print('masks_type',masks_type.shape, masks_type)
        #attention for different type
        a_type = self.attn_type(xx, masks_type)
        #print('a_type', a_type.shape, a_type)
        x = attention_types_sum(a_type, xx, masks=masks_type) 
        #print('x',x.shape, x.device)
                    
        logits = self.linear(x) # shape = (batch_size, 1)
        ##print(logits.shape)
        prob = self.sigmoid(logits)
        prob = prob.squeeze(dim = -1)
        #print('prob:', prob.shape, prob)       

        if interpret:
            return prob, xx1, a_note1, a_visit1, xx2, a_note2, a_visit2, xx3, a_note3, a_visit3, a_type, masks_type 
        else:
            return prob


class CustomModel_4codes(nn.Module):
    """
    for binary classification
    input: 4 of multiple types of codes: diag (x1), proc (x2), lab (x3), rx (x4)
    """
    def __init__(self, num_codes1, num_codes2, num_codes3, num_codes4, embedding_dim=128): 
        super(CustomModel_4codes,self).__init__() 
        #self.num_labels = num_labels 
        self.embedding1 = nn.Embedding(num_codes1, embedding_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(num_codes2, embedding_dim, padding_idx=0)
        self.embedding3 = nn.Embedding(num_codes3, embedding_dim, padding_idx=0)
        self.embedding4 = nn.Embedding(num_codes4, embedding_dim, padding_idx=0)
        self.attn_note1 = Attention(embedding_dim)
        self.attn_note2 = Attention(embedding_dim)
        self.attn_note3 = Attention(embedding_dim)
        self.attn_note4 = Attention(embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru3 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru4 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attn_visit1 = Attention(embedding_dim)
        self.attn_visit2 = Attention(embedding_dim)
        self.attn_visit3 = Attention(embedding_dim)
        self.attn_visit4 = Attention(embedding_dim)
        self.attn_type = Attention(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1) # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, masks1, x2, masks2, x3, masks3, x4, masks4, interpret = False):
        """
        Arguments:
            x: the pre-trained model embeddings of notes sequence of shape (batch_size, # visits, # notes, embedding_dim)
            masks: the padding masks of shape (batch_size, # visits, # notes) 
        """
        #Add custom layers
        #print('')
        #print('start: CustomModel_2codes')
        #print('x1:',x1.device, '; masks1:',masks1.device, '; x2:',x2.device, '; masks2:',masks2.device)
        #print('')
        
        #print('x1',x1.shape, x1)
        xx1 = self.embedding1(x1)
        #print('xx1',xx1.shape, xx1.device)
        #print('')
        a_note1 = self.attn_note1(xx1, masks1)
        #print('a_note1',a_note1.shape,a_note1.device)
        #print('')
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        #print('')
        h1, _ = self.gru1(x1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        #print('a_visit1',a_visit1.shape,a_visit1.device)
        #print('')
        x1 = attention_visits_sum(a_visit1, x1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        
        #print('x2',x2.shape, x2)
        xx2 = self.embedding2(x2)
        #print('xx2',xx2.shape, xx2)
        a_note2 = self.attn_note2(xx2, masks2)
        #print('a_note2',a_note2.shape,a_note2)
        x2 = attention_codes_sum(a_note2, xx2, masks=masks2)
        #print('x2',x2.shape, x2)
        h2, _ = self.gru2(x2)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit2 = self.attn_visit2(h2, masks2)                  #visit attention
        #print('a_visit2',a_visit2.shape,a_visit2)
        x2 = attention_visits_sum(a_visit2, x2, masks=masks2)
        #print('x2',x2.shape, x2.device)
        
        #print('x3',x3.shape, x3)
        xx3 = self.embedding3(x3)
        #print('xx3',xx3.shape, xx3.device)
        #print('')
        a_note3 = self.attn_note3(xx3, masks3)
        #print('a_note3',a_note3.shape,a_note3.device)
        #print('')
        x3 = attention_codes_sum(a_note3, xx3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        #print('')
        h3, _ = self.gru3(x3)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit3 = self.attn_visit3(h3, masks3)                  #visit attention
        #print('a_visit3',a_visit3.shape,a_visit3.device)
        #print('')
        x3 = attention_visits_sum(a_visit3, x3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        
        #print('x4',x4.shape, x4)
        xx4 = self.embedding4(x4)
        #print('xx4',xx4.shape, xx4)
        a_note4 = self.attn_note4(xx4, masks4)
        #print('a_note4',a_note4.shape,a_note4)
        x4 = attention_codes_sum(a_note4, xx4, masks=masks4)
        #print('x4',x4.shape, x4)
        h4, _ = self.gru4(x4)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit4 = self.attn_visit4(h4, masks4)                  #visit attention
        #print('a_visit4',a_visit4.shape,a_visit4)
        x4 = attention_visits_sum(a_visit4, x4, masks=masks4)
        #print('x4',x4.shape, x4.device)
               
        #stack different type of codes: diag, proc, lab, etc
        xx = torch.stack([x1, x2, x3, x4],dim = -2)
        #print('xx',xx.shape,xx)
        masks_type = torch.stack([masks1[:, 0, 0], masks2[:,0,0], masks3[:,0,0], masks4[:,0,0]], dim = 1)     #shape=(batch_size, #types)
        #print('masks_type',masks_type.shape, masks_type)
        #attention for different type
        a_type = self.attn_type(xx, masks_type)
        #print('a_type', a_type.shape, a_type)
        x = attention_types_sum(a_type, xx, masks=masks_type) 
        #print('x',x.shape, x.device)
                    
        logits = self.linear(x) # shape = (batch_size, 1)
        ##print(logits.shape)
        prob = self.sigmoid(logits)
        prob = prob.squeeze(dim = -1)
        #print('prob:', prob.shape, prob)       

        if interpret:
            return prob, xx1, a_note1, a_visit1, xx2, a_note2, a_visit2, xx3, a_note3, a_visit3, xx4, a_note4, a_visit4, a_type, masks_type 
        else:
            return prob


class CustomModel_311(nn.Module):
    """
    for binary classification
    Model for 3 type of sequential codes (diag, proc, rx), 1 type of sequential pair (lab), 1 table of static features
    Inputs: 
        num_codes1, num_codes2, num_codes3: number of codes for sequential code1 to code3
        num_codes_p1: number of codes for sequential pair
        num_static: num of static features in the table of static features
    """
    def __init__(self, num_codes1, num_codes2, num_codes3, num_codes_p1, num_static, embedding_dim=128): 
        super(CustomModel_311,self).__init__() 
        #self.num_labels = num_labels        
        #sequential features
        self.embedding1 = nn.Embedding(num_codes1, embedding_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(num_codes2, embedding_dim, padding_idx=0)
        self.embedding3 = nn.Embedding(num_codes3, embedding_dim, padding_idx=0)
        self.attn_note1 = Attention(embedding_dim)
        self.attn_note2 = Attention(embedding_dim)
        self.attn_note3 = Attention(embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru3 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attn_visit1 = Attention(embedding_dim)
        self.attn_visit2 = Attention(embedding_dim)
        self.attn_visit3 = Attention(embedding_dim)
        #sequential pairs
        self.embedding_p1 = nn.Embedding(num_codes_p1, embedding_dim, padding_idx=0)
        self.attn_note_p1 = Attention(embedding_dim+1)
        self.gru_p1 = nn.GRU(embedding_dim+1, embedding_dim+1, batch_first=True)
        self.attn_visit_p1 = Attention(embedding_dim+1)
        #
        self.attn_type = Attention(embedding_dim+1)
        #static features
        self.bn = nn.BatchNorm1d(num_static)
        self.linear_static = nn.Linear(num_static, 1)
        #
        self.linear = nn.Linear(embedding_dim + 1, 1) # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static, interpret = False):
        """
        Arguments:
            static_x: static features of shape (batch_size, num_static)
            x1, x2, x3, x4: for sequential features. Tensors of shape (batch_size, num_visits, num_notes, embedding_dim)
            masks1, masks2, masks3, masks4: padding masks of shape (batch_size, num_visits, num_notes) 
        """
        #sequential features
        #print('x1',x1.shape, x1)
        xx1 = self.embedding1(x1)
        #print('xx1',xx1.shape, xx1.device)
        #print('')
        a_note1 = self.attn_note1(xx1, masks1)
        #print('a_note1',a_note1.shape,a_note1.device)
        #print('')
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        #print('')
        h1, _ = self.gru1(x1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        #print('a_visit1',a_visit1.shape,a_visit1.device)
        #print('')
        x1 = attention_visits_sum(a_visit1, x1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        
        #print('x2',x2.shape, x2)
        xx2 = self.embedding2(x2)
        #print('xx2',xx2.shape, xx2)
        a_note2 = self.attn_note2(xx2, masks2)
        #print('a_note2',a_note2.shape,a_note2)
        x2 = attention_codes_sum(a_note2, xx2, masks=masks2)
        #print('x2',x2.shape, x2)
        h2, _ = self.gru2(x2)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit2 = self.attn_visit2(h2, masks2)                  #visit attention
        #print('a_visit2',a_visit2.shape,a_visit2)
        x2 = attention_visits_sum(a_visit2, x2, masks=masks2)
        #print('x2',x2.shape, x2.device)
        
        #print('x3',x3.shape, x3)
        xx3 = self.embedding3(x3)
        #print('xx3',xx3.shape, xx3.device)
        #print('')
        a_note3 = self.attn_note3(xx3, masks3)
        #print('a_note3',a_note3.shape,a_note3.device)
        #print('')
        x3 = attention_codes_sum(a_note3, xx3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        #print('')
        h3, _ = self.gru3(x3)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit3 = self.attn_visit3(h3, masks3)                  #visit attention
        #print('a_visit3',a_visit3.shape,a_visit3.device)
        #print('')
        x3 = attention_visits_sum(a_visit3, x3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        
        #sequential pairs
        xx_p1 = self.embedding_p1(x_key_p1) #(batch_size, #visits, #codes, embedding_dim)
        xx_p1 = torch.cat((xx_p1, x_value_p1.unsqueeze(-1)), dim = -1)   #(batch_size, #visits, #codes, embedding_dim+1)
        a_note_p1 = self.attn_note_p1(xx_p1, mask_key_p1)
        x_p1 = attention_codes_sum(a_note_p1, xx_p1, masks=mask_key_p1)
        h_p1, _ = self.gru_p1(x_p1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit_p1 = self.attn_visit_p1(h_p1, mask_key_p1)                  #visit attention
        x_p1 = attention_visits_sum(a_visit_p1, x_p1, masks=mask_key_p1)    #shape= (batch_size, embedding_dim+1). 
        #print('x_p1', x_p1.shape, x_p1.device)
        
        #pad 0
        zro = torch.zeros((len(x1), 1), dtype=torch.float).to(str(x1.device))
        x1 = torch.cat((x1, zro), dim = -1)
        #zro = torch.zeros((len(x2), 1), dtype=torch.float).to(str(x2.device))
        x2 = torch.cat((x2, zro), dim = -1)
        #zro = torch.zeros((len(x3), 1), dtype=torch.float).to(str(x3.device))
        x3 = torch.cat((x3, zro), dim = -1)
               
        #stack different type of codes: diag, proc, lab, etc
        xx = torch.stack([x1, x2, x3, x_p1],dim = -2)
        #print('xx',xx.shape,xx.device)
        masks_type = torch.stack([masks1[:, 0, 0], masks2[:,0,0], masks3[:,0,0], mask_key_p1[:,0,0]], dim = 1)     #shape=(batch_size, #types)
        #print('masks_type',masks_type.shape, masks_type)
        #attention for different type
        a_type = self.attn_type(xx, masks_type)
        #print('a_type', a_type.shape, a_type)
        x = attention_types_sum(a_type, xx, masks=masks_type)
        #print('x',x.shape, x.device)
        
        #static features
        z = self.bn(x_static)
        #print(z.shape)
        
        #combine static and squential features
        logits = self.linear(x) + self.linear_static(z)          
        #logits = self.linear(x) # shape = (batch_size, 1)
        #print(logits.shape)
        prob = self.sigmoid(logits)
        prob = prob.squeeze(dim = -1)
        #print('prob:', prob.shape, prob)       

        if interpret:
            return prob, xx1, a_note1, a_visit1, xx2, a_note2, a_visit2, xx3, a_note3, a_visit3, xx_p1, a_note_p1, a_visit_p1, a_type, masks_type, z 
        else:
            return prob
        

class CustomModel_310(nn.Module):
    """
    for binary classification
    Model for 3 type of sequential codes (diag, proc, rx), 1 type of sequential pair (lab), 0 table of static features 
    Inputs: 
        num_static: num of static features in the table of static features
    """
    def __init__(self, num_codes1, num_codes2, num_codes3, num_codes_p1, embedding_dim=128): 
        super(CustomModel_310,self).__init__() 
        #self.num_labels = num_labels        
        #sequential features
        self.embedding1 = nn.Embedding(num_codes1, embedding_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(num_codes2, embedding_dim, padding_idx=0)
        self.embedding3 = nn.Embedding(num_codes3, embedding_dim, padding_idx=0)
        self.attn_note1 = Attention(embedding_dim)
        self.attn_note2 = Attention(embedding_dim)
        self.attn_note3 = Attention(embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru3 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attn_visit1 = Attention(embedding_dim)
        self.attn_visit2 = Attention(embedding_dim)
        self.attn_visit3 = Attention(embedding_dim)
        #sequential pairs
        self.embedding_p1 = nn.Embedding(num_codes_p1, embedding_dim, padding_idx=0)
        self.attn_note_p1 = Attention(embedding_dim+1)
        self.gru_p1 = nn.GRU(embedding_dim+1, embedding_dim+1, batch_first=True)
        self.attn_visit_p1 = Attention(embedding_dim+1)
        #
        self.attn_type = Attention(embedding_dim+1)
        #static features
        #self.bn = nn.BatchNorm1d(num_static)
        #self.linear_static = nn.Linear(num_static, 1)
        #
        self.linear = nn.Linear(embedding_dim + 1, 1) # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, interpret = False):
        """
        Arguments:
            static_x: static features of shape (batch_size, num_static)
            x1, x2, x3, x4: for sequential features. Tensors of shape (batch_size, num_visits, num_notes, embedding_dim)
            masks1, masks2, masks3, masks4: padding masks of shape (batch_size, num_visits, num_notes) 
        """
        #sequential features
        #print('x1',x1.shape, x1)
        xx1 = self.embedding1(x1)
        #print('xx1',xx1.shape, xx1.device)
        #print('')
        a_note1 = self.attn_note1(xx1, masks1)
        #print('a_note1',a_note1.shape,a_note1.device)
        #print('')
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        #print('')
        h1, _ = self.gru1(x1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        #print('a_visit1',a_visit1.shape,a_visit1.device)
        #print('')
        x1 = attention_visits_sum(a_visit1, x1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        
        #print('x2',x2.shape, x2)
        xx2 = self.embedding2(x2)
        #print('xx2',xx2.shape, xx2)
        a_note2 = self.attn_note2(xx2, masks2)
        #print('a_note2',a_note2.shape,a_note2)
        x2 = attention_codes_sum(a_note2, xx2, masks=masks2)
        #print('x2',x2.shape, x2)
        h2, _ = self.gru2(x2)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit2 = self.attn_visit2(h2, masks2)                  #visit attention
        #print('a_visit2',a_visit2.shape,a_visit2)
        x2 = attention_visits_sum(a_visit2, x2, masks=masks2)
        #print('x2',x2.shape, x2.device)
        
        #print('x3',x3.shape, x3)
        xx3 = self.embedding3(x3)
        #print('xx3',xx3.shape, xx3.device)
        #print('')
        a_note3 = self.attn_note3(xx3, masks3)
        #print('a_note3',a_note3.shape,a_note3.device)
        #print('')
        x3 = attention_codes_sum(a_note3, xx3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        #print('')
        h3, _ = self.gru3(x3)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit3 = self.attn_visit3(h3, masks3)                  #visit attention
        #print('a_visit3',a_visit3.shape,a_visit3.device)
        #print('')
        x3 = attention_visits_sum(a_visit3, x3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        
        #sequential pairs
        xx_p1 = self.embedding_p1(x_key_p1) #(batch_size, #visits, #codes, embedding_dim)
        xx_p1 = torch.cat((xx_p1, x_value_p1.unsqueeze(-1)), dim = -1)
        a_note_p1 = self.attn_note_p1(xx_p1, mask_key_p1)
        x_p1 = attention_codes_sum(a_note_p1, xx_p1, masks=mask_key_p1)
        h_p1, _ = self.gru_p1(x_p1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit_p1 = self.attn_visit_p1(h_p1, mask_key_p1)                  #visit attention
        x_p1 = attention_visits_sum(a_visit_p1, x_p1, masks=mask_key_p1)
        #print('x_p1', x_p1.shape, x_p1.device)
        
        #pad 0
        zro = torch.zeros((len(x1), 1), dtype=torch.float).to(str(x1.device))
        x1 = torch.cat((x1, zro), dim = -1)
        #zro = torch.zeros((len(x2), 1), dtype=torch.float).to(str(x2.device))
        x2 = torch.cat((x2, zro), dim = -1)
        #zro = torch.zeros((len(x3), 1), dtype=torch.float).to(str(x3.device))
        x3 = torch.cat((x3, zro), dim = -1)
               
        #stack different type of codes: diag, proc, lab, etc
        xx = torch.stack([x1, x2, x3, x_p1],dim = -2)
        #print('xx',xx.shape,xx.device)
        masks_type = torch.stack([masks1[:, 0, 0], masks2[:,0,0], masks3[:,0,0], mask_key_p1[:,0,0]], dim = 1)     #shape=(batch_size, #types)
        #print('masks_type',masks_type.shape, masks_type)
        #attention for different type
        a_type = self.attn_type(xx, masks_type)
        #print('a_type', a_type.shape, a_type)
        x = attention_types_sum(a_type, xx, masks=masks_type)
        #print('x',x.shape, x.device)
        
        #static features
        #z = self.bn(x_static)
        #print(z.shape)
        
        #combine static and squential features
        #logits = self.linear(x) + self.linear_static(z)          
        logits = self.linear(x) # shape = (batch_size, 1)
        #print(logits.shape)
        prob = self.sigmoid(logits)
        prob = prob.squeeze(dim = -1)
        #print('prob:', prob.shape, prob)       

        if interpret:
            return prob, xx1, a_note1, a_visit1, xx2, a_note2, a_visit2, xx3, a_note3, a_visit3, xx_p1, a_note_p1, a_visit_p1, a_type, masks_type 
        else:
            return prob
        
class CustomModel_401(nn.Module):
    """
    for binary classification
    Model for 4 type of sequential codes (diag, proc, rx, lab), 0 type of sequential pair (lab), 1 table of static features 
    Inputs: 
        num_static: num of static features in the table of static features
    """
    def __init__(self, num_codes1, num_codes2, num_codes3, num_codes4, num_static, embedding_dim=128): 
        super(CustomModel_401,self).__init__() 
        #self.num_labels = num_labels        
        #sequential features
        self.embedding1 = nn.Embedding(num_codes1, embedding_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(num_codes2, embedding_dim, padding_idx=0)
        self.embedding3 = nn.Embedding(num_codes3, embedding_dim, padding_idx=0)
        self.embedding4 = nn.Embedding(num_codes4, embedding_dim, padding_idx=0)
        self.attn_note1 = Attention(embedding_dim)
        self.attn_note2 = Attention(embedding_dim)
        self.attn_note3 = Attention(embedding_dim)
        self.attn_note4 = Attention(embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru2 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru3 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.gru4 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.attn_visit1 = Attention(embedding_dim)
        self.attn_visit2 = Attention(embedding_dim)
        self.attn_visit3 = Attention(embedding_dim)
        self.attn_visit4 = Attention(embedding_dim)       
        #
        self.attn_type = Attention(embedding_dim)
        #static features
        self.bn = nn.BatchNorm1d(num_static)
        self.linear_static = nn.Linear(num_static, 1)
        #
        self.linear = nn.Linear(embedding_dim, 1) # load and initialize weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, masks1, x2, masks2, x3, masks3, x4, masks4, x_static, interpret = False):
        """
        Arguments:
            static_x: static features of shape (batch_size, num_static)
            x1, x2, x3, x4: for sequential features. Tensors of shape (batch_size, num_visits, num_notes, embedding_dim)
            masks1, masks2, masks3, masks4: padding masks of shape (batch_size, num_visits, num_notes) 
        """
        #sequential features
        #print('x1',x1.shape, x1)
        xx1 = self.embedding1(x1)
        #print('xx1',xx1.shape, xx1.device)
        #print('')
        a_note1 = self.attn_note1(xx1, masks1)
        #print('a_note1',a_note1.shape,a_note1.device)
        #print('')
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        #print('')
        h1, _ = self.gru1(x1)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        #print('a_visit1',a_visit1.shape,a_visit1.device)
        #print('')
        x1 = attention_visits_sum(a_visit1, x1, masks=masks1)
        #print('x1',x1.shape, x1.device)
        
        #print('x2',x2.shape, x2)
        xx2 = self.embedding2(x2)
        #print('xx2',xx2.shape, xx2)
        a_note2 = self.attn_note2(xx2, masks2)
        #print('a_note2',a_note2.shape,a_note2)
        x2 = attention_codes_sum(a_note2, xx2, masks=masks2)
        #print('x2',x2.shape, x2)
        h2, _ = self.gru2(x2)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit2 = self.attn_visit2(h2, masks2)                  #visit attention
        #print('a_visit2',a_visit2.shape,a_visit2)
        x2 = attention_visits_sum(a_visit2, x2, masks=masks2)
        #print('x2',x2.shape, x2.device)
        
        #print('x3',x3.shape, x3)
        xx3 = self.embedding3(x3)
        #print('xx3',xx3.shape, xx3.device)
        #print('')
        a_note3 = self.attn_note3(xx3, masks3)
        #print('a_note3',a_note3.shape,a_note3.device)
        #print('')
        x3 = attention_codes_sum(a_note3, xx3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        #print('')
        h3, _ = self.gru3(x3)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit3 = self.attn_visit3(h3, masks3)                  #visit attention
        #print('a_visit3',a_visit3.shape,a_visit3.device)
        #print('')
        x3 = attention_visits_sum(a_visit3, x3, masks=masks3)
        #print('x3',x3.shape, x3.device)
        
        #print('x4',x4.shape, x4)
        xx4 = self.embedding4(x4)
        #print('xx4',xx4.shape, xx4)
        a_note4 = self.attn_note4(xx4, masks4)
        #print('a_note4',a_note4.shape,a_note4)
        x4 = attention_codes_sum(a_note4, xx4, masks=masks4)
        #print('x4',x4.shape, x4)
        h4, _ = self.gru4(x4)             #shape= (batch_size, #visits, embedding_dim). 
        a_visit4 = self.attn_visit4(h4, masks4)                  #visit attention
        #print('a_visit4',a_visit4.shape,a_visit4)
        x4 = attention_visits_sum(a_visit4, x4, masks=masks4)
        #print('x4',x4.shape, x4.device)
                      
        #stack different type of codes: diag, proc, lab, etc
        xx = torch.stack([x1, x2, x3, x4],dim = -2)
        #print('xx',xx.shape,xx)
        masks_type = torch.stack([masks1[:, 0, 0], masks2[:,0,0], masks3[:,0,0], masks4[:,0,0]], dim = 1)     #shape=(batch_size, #types)
        #print('masks_type',masks_type.shape, masks_type)
        #attention for different type
        a_type = self.attn_type(xx, masks_type)
        #print('a_type', a_type.shape, a_type)
        x = attention_types_sum(a_type, xx, masks=masks_type)
        #print('x',x.shape, x.device)
        
        #static features
        z = self.bn(x_static)
        #print(z.shape)
        
        #combine static and squential features
        logits = self.linear(x) + self.linear_static(z)          
        #logits = self.linear(x) # shape = (batch_size, 1)
        #print(logits.shape)
        prob = self.sigmoid(logits)
        prob = prob.squeeze(dim = -1)
        #print('prob:', prob.shape, prob)       

        if interpret:
            return prob, xx1, a_note1, a_visit1, xx2, a_note2, a_visit2, xx3, a_note3, a_visit3, xx4, a_note4, a_visit4, a_type, masks_type, z 
        else:
            return prob

######Training model functions
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 

def train(model, train_dataset, val_dataset, batch_size, collate_fn, criterion, optimizer, n_epochs, saveDir, singleGPU=False, task='binary', continueTrain = 2):
    """
    Use Accelerate to speed up training.
    
    Arguments:
        model: the model to train
        train_dataset: training dataset
        val_dataset: validation dataset, used to stop training
        batch_size: batch size for training
        criterion: loss function
        optimizer: optimization algorithm
        n_epochs: total number of epochs
        saveDir: directory to save the models; default is the current directory
        singleGPU: True if training on a single GPU
        model_tmp: temporary model file
        task: 'binary', 'multiclass', 'regression'
        continueTrain: number of epochs to continue training after stopping criterion worsens. If training doesn't improves during continueTrain, go back to the previous best model before worsening.
    Output:
        trained model saved in saveModel
    """
    #To be used the saved model name
    num_seq_vars =train_dataset.num_seqs
    num_seqPairs_vars = train_dataset.num_seqPairs 
    num_statFeature_vars = train_dataset.num_statFeature

    print("sigleGPU:", singleGPU)
    if singleGPU:
        accelerator = Accelerator(gradient_accumulation_steps=2)
    else:
        accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)

    num_class = model.num_classes
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,drop_last=True,num_workers=16,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=16,pin_memory=True)
         
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    #accelerator.print(len(train_loader))
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # n_epochs can control the learning rate
    #lr_scheduler = get_linear_schedule_with_warmup(
    #    optimizer=optimizer,
    #    num_warmup_steps=100,
    #    num_training_steps=len(train_loader) * n_epochs*2
    #)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,  max_lr=0.001,  steps_per_epoch=len(train_loader),  epochs=n_epochs,  div_factor = 25,  final_div_factor = 10)

    #model.train()
    #auc = 0.0
    auc = -float('inf')
    counter = 0     #counter for continueTrain
    start_time0 = time.time()
    
    for epoch in range(n_epochs):
        #print(epoch)
        model.train()       
        #save previous best model
        if counter < 1:
            unwrapped_model = accelerator.unwrap_model(model)
            model_prev = copy.deepcopy(unwrapped_model)
            auc_prev = auc
        
        train_loss = 0
        i = 0
        #for x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static, y in train_loader:
        for *XsMs, y in train_loader:
            # If any of x is empty, skip the batch
            #if min(sum(masks1[:,0,0]), sum(masks2[:,0,0]),sum(masks3[:,0,0])) > 0 :
            l = len(XsMs)
            minsum = sum(XsMs[1][:,0,0])
            for m in range(1, int(l/2)):
                try:
                    minsum = min(minsum,sum(XsMs[2*m+1][:,0,0]))
                except:
                    pass
            #print(minsum)
            if minsum > 0:
                optimizer.zero_grad()
                #y_hat = model(x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static)
                y_hat = model(*XsMs)
                #print('y_hat:',y_hat.device, y_hat.shape, y_hat)
                #print('')
                #print('y:',y.shape, y)
                #loss = None
                #criterion: BCELoss (binary), CrossEntropyLoss (multi-class), MSELoss (regression)
                if task == 'binary':
                    prob = torch.sigmoid(y_hat)
                    y_hat = prob.squeeze(dim = -1)
                if task == 'multiclass':
                    y = y.long()
                if task == 'regression':
                    y_hat = y_hat.squeeze(dim = -1)
                loss = criterion(y_hat, y)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.item()
                #print('train_loss:',train_loss.device)
            if i % 500 == 0:
                accelerator.print("Batch: ", i, ". Batch Loss:", loss.item(), ". Avg training loss so far: ", train_loss / (i+1))
                #accelerator.print('Lerning rate:', get_lr(optimizer))
            i = i+1
        train_loss = train_loss / len(train_loader)
        accelerator.print('Epoch: {} \t Training Loss: {:.6f} \t Time so far (minutes): {}'.format(epoch+1, train_loss, (time.time() - start_time0)/60))  
        #accelerator.print('lerning rate:', get_lr(optimizer))
        
        #evaluation and stop training using val_dataset
        model.eval()
        y_score = []
        y_true = []
        i=0
        #for x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static, y in val_loader:
        for *XsMs, y in val_loader:
            #x =x.float()
            #y_hat = model(x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static)
            y_hat = model(*XsMs)
            if task == 'binary':
                prob = torch.sigmoid(y_hat)
                y_hat = prob.squeeze(dim = -1)
            if task == 'multiclass':
                y_hat = torch.softmax(y_hat,dim=1)  
            if task == 'regression':
                y_hat = y_hat.squeeze(dim = -1)
            #y_score.extend(y_hat.tolist())
            #y_true.extend(y.tolist())
            y_score.extend(accelerator.gather(y_hat).tolist())
            y_true.extend(accelerator.gather(y).tolist())
            if i % 500 == 0:
                accelerator.print("Batch: ", i, ". Length of y_score:", len(y_score))
            i = i + 1
        # The last thing we need to do is to truncate the predictions and labels we concatenated
        # together as the prepared evaluation dataloader has a little bit more elements to make
        # batches of the same size on each process.
        y_score = y_score[:len(val_dataset)]
        y_true = y_true[:len(val_dataset)]
        y_score = torch.tensor(y_score,dtype=torch.float32)
        #print(len(y_score))
        
        #metric_auc = load_metric('roc_auc')
        #metric_auc = evaluate.load('roc_auc')
        #auc = metric_auc.compute(prediction_scores = y_score, references=y_true)
        #auc = auc['roc_auc']
        if task == 'binary':
            y_true = torch.tensor(y_true,dtype=torch.int32)
            auc = auroc(y_score,y_true,task='binary')
        if task == 'multiclass':
            y_true = torch.tensor(y_true,dtype=torch.int32)
            auc = auroc(y_score,y_true,task='multiclass',num_classes=num_class,average='weighted',thresholds=None)
            auc_ind = auroc(y_score, y_true,task='multiclass',num_classes=num_class,average=None,thresholds=None)
        if task == 'regression':
            y_true = torch.tensor(y_true,dtype=torch.float32)
            auc = -criterion(y_score, y_true)      #should be -MSE, not auc, but just used the name
        accelerator.print('Epoch: {} \t Validation AUC or -MSE: {:.6f}'.format(epoch+1, auc))
        #print("Length of y_score:", len(y_score))
        if auc <= auc_prev:
            counter +=1
            if counter >= continueTrain or epoch  == n_epochs-1: #steps_continueTrain:
                accelerator.print("Stopping criterion (AUC or -MSE) decreases, go back to previous best model")
                model = model_prev
                auc = auc_prev
                n_epochs = epoch - counter + 1
                break
        else:
            counter = 0
    print(auc)    
    ##Save final model
    suffix = str(int(time.time()))
    if task == 'regression':
        modelFile = saveDir+"/ihanModel_"+str(num_seq_vars)+str(num_seqPairs_vars)+str(num_statFeature_vars)+"_"+str(n_epochs)+"epochs_mse"+str(round(-auc.item(), 5))+"_"+suffix+".sav"
    else:
        modelFile = saveDir+"/ihanModel_"+str(num_seq_vars)+str(num_seqPairs_vars)+str(num_statFeature_vars)+"_"+str(n_epochs)+"epochs_auc"+str(round(auc.item(), 5))+"_"+suffix+".sav"
    if singleGPU:
        torch.save(model, modelFile)
    else:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model, modelFile)
    accelerator.print('Final model:', modelFile)
    accelerator.print('Final model: Epoch = {} \t Validation AUC or - MSE = {:.6f} \t lr = {}'.format(n_epochs, auc, get_lr(optimizer)))

def train_test(model, train_dataset, val_dataset, test_datasets, batch_size, collate_fn, criterion, optimizer, 
               n_epochs, modelFile, singleGPU, aucFile, TrainLog,task='binary',continueTrain=2):
    """
    Same as train except that this one calsulate the test AUC for the final model.
    
    Arguments:
        model: the model to train
        train_dataset: training dataset
        val_dataset: validation dataset
        test_dataset: test dataset (not used in training), allow a list of multiple test_datasets
        batch_size: batch size for training
        criterion: loss function
        optimizer: optimization algorithm
        n_epochs: total number of epochs
        singleGPU: True if training on a single GPU
        modelFile: file to save the trained model
        aucFile: file to save AUC or RMSE (regression)
        task: 'binary', 'multiclass', 'regression'
        continueTrain: number of epochs to continue train after stopping criterion worsens. If training doesn't improves during continueTrain, go back to the previous best model before worsening.
    Output:
        modelFile: file to save the trained model
        aucFile: file to save AUC or RMSE (regression)
    """
    #To be used the saved model name
    num_seq_vars =train_dataset.num_seqs
    num_seqPairs_vars = train_dataset.num_seqPairs 
    num_statFeature_vars = train_dataset.num_statFeature

    print("sigleGPU:", singleGPU)
    if singleGPU:
        accelerator = Accelerator(gradient_accumulation_steps=2)
    else:
        accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)

    num_class = model.num_classes
    
    # I added drop last = True for val loader and test loader to check if that will fix the problem
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,drop_last=True,num_workers=16,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=16,pin_memory=True)   
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=16,pin_memory=True)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
       
    #accelerator.print(len(train_loader))
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # n_epochs can control the learning rate
    #lr_scheduler = get_linear_schedule_with_warmup(
    #    optimizer=optimizer,
    #    num_warmup_steps=100,
    #    num_training_steps=len(train_loader) * n_epochs*2
    #)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,  max_lr=0.001,  steps_per_epoch=len(train_loader),  epochs=n_epochs,  div_factor = 25,  final_div_factor = 10)

    #model.train()
    #auc = 0.0
    auc = -float('inf')
    counter = 0     #counter for continueTrain
    start_time0 = time.time()
    
    for epoch in range(n_epochs):
        #print(epoch)
        model.train()       
        #save previous best model
        if counter < 1:
            unwrapped_model = accelerator.unwrap_model(model)
            model_prev = copy.deepcopy(unwrapped_model)
            auc_prev = auc
        
        train_loss = 0
        i = 0
        #for x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static, y in train_loader:
        for *XsMs, y in train_loader:
            # If any of x is empty, skip the batch
            #if min(sum(masks1[:,0,0]), sum(masks2[:,0,0]),sum(masks3[:,0,0])) > 0 :
            l = len(XsMs)
            minsum = sum(XsMs[1][:,0,0])
            for m in range(1, int(l/2)):
                try:
                    minsum = min(minsum,sum(XsMs[2*m+1][:,0,0]))
                except:
                    pass
            #print(minsum)
            if minsum > 0:
                optimizer.zero_grad()
                #y_hat = model(x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static)
                y_hat = model(*XsMs)
                #print('y_hat:',y_hat.device, y_hat.shape, y_hat)
                #print('')
                #print('y:',y.shape, y)
                #loss = None
                #criterion: BCELoss (binary), CrossEntropyLoss (multi-class), MSELoss (regression)
                if task == 'binary':
                    prob = torch.sigmoid(y_hat)
                    y_hat = prob.squeeze(dim = -1)
                if task == 'multiclass':
                    y = y.long()
                if task == 'regression':
                    y_hat = y_hat.squeeze(dim = -1)
                loss = criterion(y_hat, y)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.item()
                #print('train_loss:',train_loss.device)
            if i % 500 == 0:
                accelerator.print("Batch: ", i, ". Batch Loss:", loss.item(), ". Avg training loss so far: ", train_loss / (i+1))
                #accelerator.print('Lerning rate:', get_lr(optimizer))
            i = i+1
        train_loss = train_loss / len(train_loader)
        accelerator.print('Epoch: {} \t Training Loss: {:.6f} \t Time so far (minutes): {}'.format(epoch+1, train_loss, (time.time() - start_time0)/60))  
        accelerator.print('lerning rate:', get_lr(optimizer))
        
        #evaluation and stop training using val_dataset
        model.eval()
        y_score = []
        y_true = []
        i=0
        #for x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static, y in val_loader:
        for *XsMs, y in val_loader:
            #x =x.float()
            #y_hat = model(x1, masks1, x2, masks2, x3, masks3, x_key_p1, mask_key_p1, x_value_p1, x_static)
            y_hat = model(*XsMs)
            if task == 'binary':
                prob = torch.sigmoid(y_hat)
                y_hat = prob.squeeze(dim = -1)
            if task == 'multiclass':
                y_hat = torch.softmax(y_hat,dim=1)  
            if task == 'regression':
                y_hat = y_hat.squeeze(dim = -1)
            #y_score.extend(y_hat.tolist())
            #y_true.extend(y.tolist())
            y_score.extend(accelerator.gather(y_hat).tolist())
            y_true.extend(accelerator.gather(y).tolist())
            if i % 500 == 0:
                accelerator.print("Batch: ", i, ". Length of y_score:", len(y_score))
            i = i + 1
        # The last thing we need to do is to truncate the predictions and labels we concatenated
        # together as the prepared evaluation dataloader has a little bit more elements to make
        # batches of the same size on each process.
        y_score = y_score[:len(val_dataset)]
        y_true = y_true[:len(val_dataset)]
        y_score = torch.tensor(y_score,dtype=torch.float32)
        #print(len(y_score))
        
        if task == 'binary':
            y_true = torch.tensor(y_true,dtype=torch.int32)
            auc = auroc(y_score,y_true,task='binary')
        if task == 'multiclass':
            y_true = torch.tensor(y_true,dtype=torch.int32)
            auc = auroc(y_score,y_true,task='multiclass',num_classes=num_class,average='weighted',thresholds=None)
            auc_ind = auroc(y_score, y_true,task='multiclass',num_classes=num_class,average=None,thresholds=None)
        if task == 'regression':
            y_true = torch.tensor(y_true,dtype=torch.float32)
            auc = -criterion(y_score, y_true)      #should be -MSE, not auc, but just used the name
        accelerator.print('Epoch: {} \t Validation AUC or -MSE: {:.6f}'.format(epoch+1, auc))
        #print("Length of y_score:", len(y_score))
        if auc <= auc_prev:
            counter +=1
            if counter >= continueTrain or epoch  == n_epochs-1: #steps_continueTrain:
                accelerator.print("Stopping criterion (AUC or -MSE) decreases, go back to previous best model")
                model = model_prev
                auc = auc_prev
                n_epochs = epoch - counter + 1
                break
        else:
            counter = 0
    
    time_train = (time.time() - start_time0)/60
        
    ##Save final model
    if singleGPU:
        torch.save(model, modelFile)
    else:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model, modelFile)
    accelerator.print('Final model:', modelFile)
    accelerator.print('Final model: Epoch = {} \t Validation AUC or -MSE = {:.6f} \t lr = {}'.format(n_epochs, auc, get_lr(optimizer)))
    
    #Final model AUC on test data (never used in training)
    model.eval()
    i=0
    auc_tests ={}
    auc_tests_ind ={}
    for t in range(len(test_datasets)):
        y_score = []
        y_true = []
        test_dataset = test_datasets[t]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=16,pin_memory=True)
        test_loader = accelerator.prepare(test_loader)
        for *XsMs, y in test_loader:
            y_hat = model(*XsMs, interpret = False)
            if task == 'binary':
                prob = torch.sigmoid(y_hat)
                y_hat = prob.squeeze(dim = -1)
            if task == 'multiclass':
                y_hat = torch.softmax(y_hat,dim=1)  
            if task == 'regression':
                y_hat = y_hat.squeeze(dim = -1)
            y_score.extend(accelerator.gather(y_hat).tolist())
            y_true.extend(accelerator.gather(y).tolist())
            if i % 500 == 0:
                accelerator.print("Batch: ", i, ". Length of y_score:", len(y_score))
            i = i + 1
        y_score = y_score[:len(test_dataset)]
        y_true = y_true[:len(test_dataset)]

        auc_test = eval_measure(y_score, y_true, task=task, num_classes=num_class)
    
        if task == 'multiclass':
            auc_test_ind = auc_test[1].numpy()
            auc_test = round(auc_test[0].item(),6)
        else:
            auc_test_ind = 'NaN'
            auc_test = round(auc_test.item(),6)
            
        key="test_auc"+str(t)
        auc_tests[key]=auc_test
        key="test_auc_ind"+str(t)
        auc_tests_ind[key]=auc_test_ind
        
        accelerator.print('AUC (classification) or RMSE (regression) for test_dataset {0}: {1}'.format(t, auc_test))
        
    #write out the AUCs
    AUC = {'n_epoch': n_epochs, 'time':time_train, 'task':task, 'batch_size':batch_size, 'auc_val': round(auc.item(),6)}   
    AUC = {**AUC, **auc_tests, **auc_tests_ind}
    AUC.update(TrainLog)
    accelerator.print(AUC, aucFile)
    
    aucfile = Path(aucFile)
    header = list(AUC.keys())
    print(aucfile )
    if aucfile.is_file():
        file = open(aucfile, 'a', newline ='')
        with file:   
            writer = csv.DictWriter(file, fieldnames = header)
            writer.writerow(AUC)
    else: 
        file = open(aucfile, 'w', newline ='')
        with file:    
            writer = csv.DictWriter(file, fieldnames = header)
            writer.writeheader()
            writer.writerow(AUC)
    
    accelerator.free_memory()
    
    
######Prediction, evaluation and interpretation
def tensorToList_masks(x, masks):
    """
    To convert tensor x to list but only keep if masks not 0
    x: tensor
    """
    #x = x * masks
    xList = x.tolist()
    xList1 = xList.copy()
    mList = masks.tolist()
    for i_patient, patient in enumerate(xList):
        for j_visit, visit in enumerate(patient):
            #xList1[i_patient][j_visit] = list(code for code in visit if code != 0)
            xList1[i_patient][j_visit] = list(code for code, mask in zip(visit, mList[i_patient][j_visit]) if mask)
        xList1[i_patient] = list(visit for visit in patient if len(visit)>0)
    return xList1

def contributionCoef(type, mask, x_emb, a_note, a_visit, a_type, W):
    """
    Calculate the contribution coefficients for a seqCode or seqPair
    Works for either binary or multi-class label.
    type: int, index for type in a_type
    output:
        contributionList[class][mcid][visit][code]
    """
    #rescale a_visit
    a_visit_rescaled = attention_visits_rescale(a_visit, masks=mask)
    #should rescale a_type?
    G = W.shape[-2]      # G = Num of classes for multi-class, = 1 for binary
    if G > 1:
        contributionList =[]
        for g in range(G):
            contribution = a_type[:,type,None]*a_note.squeeze(dim=-1) * a_visit_rescaled * (torch.sum(W[:,:,g,:] * x_emb, dim = -1))
            #print(g, contribution.shape)
            contributionList.append(tensorToList_masks(contribution, mask))
            #print(g, len(contributionList), len(contributionList[0]), len(contributionList[0][0]), len(contributionList[0][0][0]))
    else:
        contribution = a_type[:,type,None]*a_note.squeeze(dim=-1) * a_visit_rescaled * (torch.sum(W * x_emb, dim = -1))
        contributionList = tensorToList_masks(contribution, mask)
    
    return contributionList

def contributionCoef_stat(z, W_stat):
    """
    Calculate the contribution coefficients for static features
    """
    G = W_stat.shape[-2]
    if G > 1:
        contribution_stat =[]
        for g in range(G):
            contribution = W_stat[g,:] * z
            contribution_stat.append(contribution)   
    else:
        contribution_stat = W_stat * z
    return contribution_stat

              
def extendList(contriList, contributionsList, num_seqs, G):
    """
    Extend contriList by contributionsList for each class and type
    contriList: list(list(list(list))) if G = 1, type, mcid, visit, code
                or list(list(list(list(list)))) if G > 1, type, class, mcid, visit, code
    contributionsList: same structure
    """
    if G > 1:
        for i in range(num_seqs):
            [contriList[i][g].extend(contributionsList[i][g]) for g in range(G)]
    else:   
        [contriList[i].extend(contributionsList[i]) for i in range(num_seqs)]            
            
def catList(contriStat, contribution_stat, G):
    """
    concat contriStat and contribution_stat for each class
    contrStat: tensor if G = 1, or list(tensor) is G > 1
    """
    if G > 1:
        for g in range(G):
            contriStat[g] = torch.cat((contriStat[g], contribution_stat[g]),dim = 0)
    else:
        #print(contriStat.shape, contribution_stat.shape)
        contriStat = torch.cat((contriStat, contribution_stat),dim = 0)
    return contriStat

def eval_measure(y_score, y_obs, task='binary', num_classes=1):
    """
    Calculate the evaluation measures: auc for classification, and rmse for regression
    y_score, y_obs: list of predictions and the observed values.
    """
    #print(type(y_score), type(y_obs))
    y_score = torch.tensor(y_score,dtype=torch.float32)
    if task == 'binary':
        y_obs = torch.tensor(y_obs,dtype=torch.int32)
        auc = auroc(y_score,y_obs,task='binary')
        return auc
    elif task == 'multiclass':
        y_obs = torch.tensor(y_obs,dtype=torch.int32)
        auc = auroc(y_score,y_obs,task='multiclass',num_classes=num_classes,average='weighted',thresholds=None)
        auc_ind = auroc(y_score, y_obs,task='multiclass',num_classes=num_classes,average=None,thresholds=None)
        return auc, auc_ind
    elif task == 'regression':
        y_obs = torch.tensor(y_obs,dtype=torch.float32)
        mse = nn.MSELoss()
        #rmse = mse(y_score, y_obs)
        rmse = torch.sqrt(mse(y_score, y_obs))
        return rmse
                           

def add_nested_lists(list1, list2, w1, w2):
    """
    calculate w1 * list1 + w2 * list2 elementwise. list1 and list2 can be list(list(...))) of any depth
    """
    if isinstance(list1, list) and list1 and isinstance(list2, list) and list2:
        if isinstance(list1[0], list) or isinstance(list2[0], list):
            return [add_nested_lists(a, b, w1, w2) for a, b in zip(list1, list2)]
        else:
            output = [w1 * a + w2 * b for a, b in zip(list1, list2)]
            return output
    else:
        return []

def pred(model, df, ID = 'mcid', tgt = None, cols_seqCode = ['datList1','datList2','datList4'],
                             cols_seqPair = ['labPairList'],cols_statFeature = ['num_age','num_genderm'],
                       batch_size=32, interpret = False,task = 'binary',evaluation=False):
    """
    Predcition and/or interpretation for the given model
    Arguments:
        model: model to be used for prediction and interpretation
        df: data frame on which to perform prediction. Should have columns listed in cols_seqCode,cols_seqPair,cols_statFeature
        tgt: column name for target/label, if no label, tgt = None
        cols_seqCode: column names for sequenctial medical codes: diagnosis, procedure, lab, medication. At least one should occur.
        cols_seqPair: column names for sequenctial pair data such as lab test: loinc_cd: test_value. Maybe None
        cols_statFeature: column names for static features such as age, gender. Maybe None
        batch_size: number of samples in a batch
        interpret: if True, both prediction and interpretation are performed. If False, only predcition is done.
    output:
        y_score, *contriList_seq, *contriList_seqPair, *xLists, contrStat, xStat, (y_obs) if intepret = True
        y_score, (y_obs) if interpret = False
    """       
    num_seqs = 0 if cols_seqCode is None else len(cols_seqCode)
    num_seqPairs = 0 if cols_seqPair is None else len(cols_seqPair)
    statFeature = False if (cols_statFeature is None) or (len(cols_statFeature) == 0) else True
    print(num_seqs, num_seqPairs, statFeature)

    dataset = CustomDataset(df, tgt = tgt, cols_seqCode = cols_seqCode, cols_seqPair = cols_seqPair,cols_statFeature = cols_statFeature)
    collate_fn_arg = partial(collate_fn, num_seq_vars=num_seqs, num_seqPairs_vars=num_seqPairs, statFeature=statFeature, tgt= tgt, max_ncodes_perVisit = 512)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_arg, shuffle = False)
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    
    G = params['linear.weight'].shape[0]        #number of target classes
    if 'linear_mh.weight' in params.keys():
        H = params['linear_mh.weight'].shape[1]     #number of heads
    else:
        H = 1
    print("number of heads:", H)

    y_score = []
    y_obs = []
    ids = []
    if G > 1:
        contriList_seq = [[[] for i in range(G)] for j in range(num_seqs)]
    else:
        contriList_seq = [[] for i in range(num_seqs)]
    if num_seqPairs > 0:
        if G > 1:
            contriList_seqPair = [[[] for i in range(G)] for j in range(num_seqs, num_seqs + num_seqPairs)]
        else:
            contriList_seqPair = [[] for i in range(num_seqs, num_seqs + num_seqPairs)]
    xLists = [[] for i in range(num_seqs + num_seqPairs)]
    if statFeature:
        device = accelerator.device
        if G > 1:
            contrStat = [torch.empty(0).to(device) for i in range(G)]
        else:
            contrStat = torch.empty(0).to(device)
        xStat = torch.empty(0).to(device)
    
    i = 0
    if interpret:
        for dataBatch in data_loader: 
            if tgt is None:
                XsMs = dataBatch
            else:
                *XsMs, y = dataBatch
                y_obs.extend(y.tolist())
            #print('batch:', i)

            y_hat, t1, t2,  a_type, masks_type, z   = model(*XsMs, interpret = True) 

            if task == 'binary':
                prob = torch.sigmoid(y_hat)
                y_hat = prob.squeeze(dim = -1)
            if task == 'multiclass':
                y_hat = torch.softmax(y_hat,dim=1)  
            if task == 'regression':
                y_hat = y_hat.squeeze(dim = -1)
            #params needed : type, (mask, x_emb, a_note, a_visit), a_type, W
            #model results t1, t2 are tuple of tuple: ((x_p1, mask_key_p1, (xx_p1, a_note_p1, a_visit_p1))    
            #print(a_type)
            y_score.extend(y_hat.tolist())      
            if H > 1:
                W_mh = params['linear_mh.weight'][0]
            if num_seqPairs > 0:
                W = params['linear.weight'][:,:-1][None,None,:,:]
                W_seqPair = params['linear.weight'][None,None,:,:]
                # t1[:][1] is the mask, returned directly from model input param 
                for h in range(H):
                    tt2 = t2[h]         
                    if h == 0:
                        contributionsList_seqPair = [contributionCoef(j+num_seqs, tt2[j][1], tt2[j][2],tt2[j][3], tt2[j][4],a_type[h], W_seqPair) for j in range(num_seqPairs)]   #for 1st head
                    else:
                        contributionsList_seqPair_h = [contributionCoef(j+num_seqs, tt2[j][1], tt2[j][2],tt2[j][3], tt2[j][4],a_type[h], W_seqPair) for j in range(num_seqPairs)]   #for each extra head
                        if h == 1:
                            contributionsList_seqPair = add_nested_lists(contributionsList_seqPair, contributionsList_seqPair_h, W_mh[h-1].item(), W_mh[h].item())
                        else: 
                            contributionsList_seqPair = add_nested_lists(contributionsList_seqPair, contributionsList_seqPair_h, 1.0, W_mh[h].item())
                #print(len(contributionsList_seqPair))
                extendList(contriList_seqPair, contributionsList_seqPair, num_seqPairs, G)
            else:
                W = params['linear.weight'][None,None,:,:]
            for h in range(H):
                tt1 = t1[h] 
                if h == 0:
                    contributionsList_seq = [contributionCoef(j, tt1[j][1], tt1[j][2],tt1[j][3], tt1[j][4],a_type[h], W) for j in range(num_seqs)]
                    #print(h, contributionsList_seq[0][0][0])
                else: 
                    contributionsList_seq_h = [contributionCoef(j, tt1[j][1], tt1[j][2],tt1[j][3], tt1[j][4],a_type[h], W) for j in range(num_seqs)]
                    #print(h, contributionsList_seq_h[0][0][0])
                    if h == 1:
                        contributionsList_seq = add_nested_lists(contributionsList_seq, contributionsList_seq_h, W_mh[h-1].item(), W_mh[h].item())
                    else:
                        contributionsList_seq = add_nested_lists(contributionsList_seq, contributionsList_seq_h, 1.0, W_mh[h].item())
                    #print(h, contributionsList_seq[0][0][0])
            extendList(contriList_seq, contributionsList_seq, num_seqs, G)
            #print(len(contriList_seq), len(contriList_seq[0]), len(contriList_seq[0][0]), len(contriList_seq[0][0][0]),len(contriList_seq[0][0][0][0]))

            #[xLists[i].extend(tensorToList_masks(XsMs[2*i], XsMs[2*i+1])) for i in range(num_seqs+num_seqPairs)]
            [xLists[i].extend(tensorToList_masks(XsMs[2*i], XsMs[2*i+1])) for i in range(num_seqs)] 
            [xLists[i+num_seqs].extend(tensorToList_masks(XsMs[3*i+2*num_seqs], XsMs[3*i+1+2*num_seqs])) for i in range(num_seqPairs)] 
   
            if statFeature:
                W_stat = params['linear_static.weight']
                #contribution_stat = W_stat * z
                #contrStat = torch.cat((contrStat, contribution_stat),dim = 0)
                contribution_stat = contributionCoef_stat(z, W_stat)
                contrStat = catList(contrStat, contribution_stat, G)
                xStat = torch.cat((xStat, XsMs[-1]),dim = 0)
            
            if i % 100 == 0:
                print("Batch: ", i, ". Length of y_score:", len(y_score))
            i = i + 1
        #return y_score, contrList1, xList1, contrList2, xList2,contrList3, xList3, contrList_p1, xList_p1, contrStat, xStat, y_obs
        #print(len(contriList_seq), len(contriList_seq[0]), len(contriList_seq[0][0]), len(contriList_seq[0][0][0]),len(contriList_seq[0][0][0][0]))
        #print(len(xLists), len(xLists[0]), len(xLists[0][0]), len(xLists[0][0][0]))
        
        if tgt is None:
            if num_seqPairs > 0:
                if not statFeature:
                    return y_score, *contriList_seq, *contriList_seqPair, *xLists
                else:
                    return y_score, *contriList_seq, *contriList_seqPair, *xLists, contrStat, xStat
            else:
                if not statFeature:
                    return y_score, *contriList_seq, *xLists
                else:
                    return y_score, *contriList_seq, *xLists, contrStat, xStat
        else:
            if evaluation:   
                measure = eval_measure(y_score, y_obs, task=task, num_classes=G)
                accelerator.print('AUC (classification) or RMSE (regression):',  measure)    
            if num_seqPairs > 0:
                if not statFeature:
                    return y_score, *contriList_seq, *contriList_seqPair, *xLists, y_obs
                else:
                    return y_score, *contriList_seq, *contriList_seqPair, *xLists, contrStat, xStat, y_obs
            else:
                if not statFeature:
                    return y_score, *contriList_seq, *xLists, y_obs
                else:
                    return y_score, *contriList_seq, *xLists, contrStat, xStat, y_obs
    else:
        ids.extend(df[ID].tolist())
        for dataBatch in data_loader: 
            if tgt is None:
                XsMs = dataBatch
            else:
                *XsMs, y = dataBatch
                y_obs.extend(y.tolist())
                
            y_hat = model(*XsMs, interpret = False)
            if task == 'binary':
                prob = torch.sigmoid(y_hat)
                y_hat = prob.squeeze(dim = -1)
            if task == 'multiclass':
                y_hat = torch.softmax(y_hat,dim=1)  
            if task == 'regression':
                y_hat = y_hat.squeeze(dim = -1)
            y_score.extend(y_hat.tolist())
            if i % 1000 == 0:
                print("Batch: ", i, ". Length of y_score:", len(y_score))
            i = i + 1
        if tgt is None:
            return ids, y_score
        else:
            if evaluation:   
                measure = eval_measure(y_score, y_obs, task=task, num_classes=G)
                accelerator.print('AUC (classification) or RMSE (regression):',  measure)   
            return ids, y_score, y_obs
        
def pred_datainchunks(model, dataFile, chunksize, ID, tgt,cols_seqCode,cols_seqPair,cols_statFeature,batch_size,interpret,task,
                      dos_cols, med_type, dos_cols_p,med_type_p,cols_pairList,evaluation): 
    """
    Perform predictions on input data in chunks, with the option to interpret the results.
    If 'interpret' is set to True, it returns the predicted data along with summary statistics.
    If 'interpret' is set to False, it only returns the predicted score.

    Arguments:
        model (nn.Module): PyTorch model for prediction.
        dataFile (str): Path to dataset.
        chunksize (int): Number of rows to read at one time (to form a chunk). 
        ID (str or int): ID to distinguish samples.
        tgt (str): The target variable column name in dataset.
        cols_seqCode (List[str]): List of sequential codes column names in dataset. 
        cols_seqPair (List[str]): List of sequential pair column names in dataset.
        cols_statFeature (List[str]): List of static features column names in dataset.
        batch_size (int): Size of batch for prediction.
        interpret (bool): If set to True, enables interpretation of results.
        task (str): Task for which model was trained.
        dos_cols (str): Column name for Date of Service in dataset.
        diag_type (str): Type of diagnosis in dataset.
        cols_stat (str): Other statistical column names in dataset. 

    Returns:
        DataFrame or Tuple of DataFrame: 
            If 'interpret' is True, it returns a tuple of three DataFrames - predicted data, sum of contribution, summary of contribution by ID and code.
            If 'interpret' is False, it returns a single DataFrame of predicted score.          
    """
    df_data=[]
    df_sum = []
    df_stat = []
    df_final = pd.DataFrame()
    df_sum_final = pd.DataFrame()
    df_mean_final = pd.DataFrame()
 
    if interpret:
        if chunksize is not None:
            reader = pd.read_csv(dataFile,chunksize = chunksize)
            for i ,chunk in enumerate(reader):
                print(f"processing chunk {i}")
                df_contribution,df_contribution_mcid_code_sum,df_contribution_code_summary=pred_interpret(model,chunk,ID, tgt,
                   cols_seqCode,cols_seqPair,cols_statFeature,batch_size,interpret,task,
                   dos_cols, med_type, dos_cols_p,med_type_p,cols_pairList,evaluation)
                #df_contribution_code_summary.to_csv(f'mean_chunk_{i}.csv',index=False)
                df_data.append(df_contribution)
                df_sum.append(df_contribution_mcid_code_sum)
                df_stat.append(df_contribution_code_summary)
        else:
            chunk = pd.read_csv(dataFile)
            df_contribution,df_contribution_mcid_code_sum,df_contribution_code_summary=pred_interpret(model,chunk,ID, tgt,cols_seqCode,cols_seqPair,
                   cols_statFeature,batch_size,interpret,task,
                   dos_cols, med_type, dos_cols_p,med_type_p,cols_pairList,evaluation)
            df_contribution_code_summary.to_csv(f'mean_code.csv',index=False)
            df_data.append(df_contribution)
            df_sum.append(df_contribution_mcid_code_sum)
            df_stat.append(df_contribution_code_summary)

        df_final = pd.concat(df_data,ignore_index=True)        #columns: ['mcid', 'dos', 'type', 'codeIndex', 'y_obs', 'y_score_0', ..., 'contribCoef_0', ...]
        df_sum_final = pd.concat(df_sum,ignore_index=True)     #['mcid', 'y_obs', 'y_score_0', ...,'type', 'codeIndex', 'dos_count', 'contribCoef_0', ...]
        df_mean_final = pd.concat(df_stat,ignore_index=True)   #['type', 'codeIndex', 'count', 'contribCoef_0_sum', ...]
        
        #print(df_contribution_mcid_code_sum[df_contribution_mcid_code_sum.type == 'static'].head())
        #Agg across all chunks
        #if len(df_stat) > 1:
        df_mean_final = df_mean_final.groupby(['type','codeIndex'],dropna=False).sum().reset_index()
        #print(df_mean_final[df_mean_final.type == 'static'].head())
        cols_contribCoef = [x for x in df_mean_final.columns if 'contribCoef' in x]
        for col in cols_contribCoef:
            df_mean_final[col] = df_mean_final[col]/df_mean_final['count']
        cols = [x.replace('_sum', '_mean') for x in df_mean_final.columns]
        df_mean_final.columns = cols

        print('df_contribution shape :' ,df_final.shape)
        print('df_contribution_mcid_code_sum shape :' ,df_sum_final.shape)
        print('df_contribution_code_summary shape : ' ,df_mean_final.shape)

        return  df_final,df_sum_final,df_mean_final
    else:
        df = pd.read_csv(dataFile)
        mcid,y_score,y_obs = pred(model, df, ID, tgt, cols_seqCode , cols_seqPair,cols_statFeature,batch_size,interpret,task,evaluation)
        if y_obs == None:
            pred_df = pd.DataFrame({'mcid':mcid,'y_score':y_score})
        else:
            pred_df = pd.DataFrame({'mcid':mcid,'y_score':y_score,'y_obs':y_obs})
            
        return pred_df
    
def pred_interpret(model,chunk,ID, tgt,cols_seqCode,cols_seqPair,cols_statFeature,batch_size,interpret,task,
                   dos_cols, med_type, dos_cols_p,med_type_p,cols_pairList,evaluation):
    """
    Handles different model combinations based on the presence of sequence pairs and statistical features and the number of sequential codes.
    Performs predictions using the given model, then interprets the results and generates summary statistics.

    Arguments:
        model (nn.Module): PyTorch model for prediction.
        chunk (DataFrame): Chunk of dataset for prediction.
        tgt (str): The target variable column name in the dataset.
        cols_seqCode (List[str]): List of sequential codes column names in the dataset.
        cols_seqPair (List[str]): List of sequential pair column names in the dataset.
        cols_statFeature (List[str]): List of statistical features column names in the dataset.
        batch_size (int): Size of batch for prediction.
        interpret (bool): If set to True, enables interpretation of results.
        task (str): Task for which the model was trained.
        dos_cols (str): Columns name for Date of Service in dataset.
        labPair (str): Column name for label pair in dataset.
        static (str): Static column name in data set.
        diag_type (str): Type of diagnosis in dataset.
        cols_stat (str): Other statistical column names in dataset.

    Returns:
        Tuple of DataFrame: It returns a tuple of three DataFrame -
                            DataFrame of prediction contributions,
                            DataFrame with sum of contributions,
                            and summary of contribution by mcid and code.         
    """
    num_seqs = 0 if cols_seqCode is None else len(cols_seqCode)
    num_seqPairs = 0 if cols_seqPair is None else len(cols_seqPair)
    statFeature = False if (cols_statFeature is None) or (len(cols_statFeature) == 0) else True
    
    if tgt is None:
        y_score,*pred_contr_outputs=pred(model,chunk, ID, tgt ,cols_seqCode, cols_seqPair,cols_statFeature,batch_size,interpret,task,evaluation)
        y_obs = None
    else:
        y_score,*pred_contr_outputs,y_obs=pred(model,chunk, ID, tgt ,cols_seqCode, cols_seqPair,cols_statFeature,batch_size,interpret,task,evaluation)
        
    #intializing variables for contribution output variables
    contrLists = []
    xlists = []
    contrLists_p = []
    contrStat = []
    xStat = []
    
    if (num_seqPairs > 0) and statFeature:
        xStat=(pred_contr_outputs[-1])
        contrStat=(pred_contr_outputs[-2])
        for i in range(num_seqs):
            contrLists.append(pred_contr_outputs[i])
            xlists.append(pred_contr_outputs[num_seqs+num_seqPairs+i])
        for i in range(num_seqPairs):
            contrLists_p.append(pred_contr_outputs[num_seqs +i])
            xlists.append(pred_contr_outputs[2* num_seqs+num_seqPairs+i])
    elif (num_seqPairs > 0):
        for i in range(num_seqs):
            contrLists.append(pred_contr_outputs[i])
            xlists.append(pred_contr_outputs[num_seqs+num_seqPairs+i])
        for i in range(num_seqPairs):
            contrLists_p.append(pred_contr_outputs[num_seqs +i])
            xlists.append(pred_contr_outputs[2* num_seqs+num_seqPairs+i])
    elif statFeature:
        xStat=(pred_contr_outputs[-1])
        contrStat=(pred_contr_outputs[-2])
        for i in range(num_seqs):
            contrLists.append(pred_contr_outputs[i])
            xlists.append(pred_contr_outputs[num_seqs+num_seqPairs+i])
    else:
        for i in range(num_seqs):
            contrLists.append(pred_contr_outputs[i])
            xlists.append(pred_contr_outputs[num_seqs+i])
            
    num_xlists = len(xlists)
    print(num_seqs, num_xlists,num_seqPairs, statFeature)
            
    df_contribution,df_contribution_mcid_code_sum,df_contribution_code_summary = interpretation(chunk,ID, y_obs,y_score,
                    dos_cols,med_type,contrLists, xlists, 
                    dos_cols_p,med_type_p,contrLists_p, cols_pairList,
                    contrStat,xStat,cols_statFeature, task)

    return df_contribution,df_contribution_mcid_code_sum,df_contribution_code_summary
    

def interpretation(df,ID, y_obs,y_score, 
                   dos_cols,med_type,contrLists, xlists, 
                   dos_cols_p,med_type_p,contrLists_p, cols_pairList,
                   contrStat,xStat,cols_statFeature, task):
    """
    Interprets the prediction results, formats the output, and performs summary statistics.
    The function can handle various model combinations based on the presence of the sequence pairs,
    statistical features, and the number of sequential codes.

    Arguments:
        df (pd.DataFrame): The input data frame containing the data.
        ID (str or int): Identifier for each data sample.
        y_obs (list or array): The observed/true values.
        y_score (list or array): The predicted values from the model.
        dos_cols (list): The column names referring to the Date of Service.
        med_type (str): The type of medical conditions.
        contrLists (list): The list of contribution values.
        xlists (list): The list of X-values corresponding to 'contrLists'.
        dos_cols_p (list): The column names referring to the Date of Service for sequence pairs.
        med_type_p (str): The type of medical conditions for sequence pairs.
        contrLists_p (list): The list of contribution values for sequence pairs.
        cols_pairList (list): List of column names for sequence pair.
        contrStat (list): The list of contribution values for statistical features.
        xStat (list): The list of X-values corresponding to 'contrStat'.
        cols_statFeature (list): List of column names for statistical features.
        task (str): The task for which the model was trained.

    Returns:
        df_contribution (pd.DataFrame): Data frame containing contributions of each feature.
        df_contribution_sum (pd.DataFrame): Data frame containing sum of contributions for each feature.
        df_contribution_mcid_code_summary (pd.DataFrame): Data frame containing mean of the contributions, grouped by ID and code index.        

    """
    num_seqs = len(med_type)
    num_seqPairs = len(med_type_p)
    statFeature = False if (cols_statFeature is None) or (len(cols_statFeature) == 0) else True

    mcidList = df[ID].tolist()
    df_contribution = pd.DataFrame() 

    for i, col in enumerate(dos_cols):
        dos_list_name = f'dosList{i + 1}'
        dos_list = df[col].apply(eval).tolist()
        locals()[dos_list_name] = dos_list
        df_contribution_name = f'df_contribution{i + 1}'
        df_contribution_format = contribution_reformat(mcidList, y_obs, y_score, dos_list, contrLists[i], xlists[i], type=med_type[i], task=task)
        locals()[df_contribution_name] = df_contribution_format
            
        df_contribution = pd.concat([df_contribution, locals()[df_contribution_name]], axis=0)
        #print(df_contribution.columns)
   
    if num_seqPairs > 0:
        for i, col in enumerate(dos_cols_p):
            dos_list_name = f'dosList_p{i + 1}'
            dos_list = df[col].apply(eval).tolist()
            locals()[dos_list_name] = dos_list
            df_contribution_name = f'df_contribution_p{i + 1}'
            pairList = df[cols_pairList[i]].apply(eval).tolist()
            df_contribution_format = contribution_seqPair_reformat(mcidList, y_obs, y_score, dos_list, contrLists_p[i],pairList, type=med_type_p[i],task = task) 
            locals()[df_contribution_name] = df_contribution_format
            
            df_contribution = pd.concat([df_contribution, locals()[df_contribution_name]], axis=0)            
        
    if statFeature:
        df_contribution_stat = contribution_static_reformat(mcidList,y_obs, y_score,contrStat, xStat, cols_statFeature,task = task)
        df_contribution = pd.concat([df_contribution, df_contribution_stat],axis = 0)
        
    df_contribution['dos'] = df_contribution['dos'].astype(str)
    #print(df_contribution.columns)

    #Sum over the same code for each visit
    groupby_cols = [col for col in df_contribution.columns if all(x not in col for x in ['dos', 'contribCoef', 'value'])]
    agg_cols = [col for col in df_contribution.columns if any(x in col for x in ['dos', 'contrib'])]
    agg_dict = {col: 'sum' for col in agg_cols if 'dos' not in col}
    agg_dict['dos'] = 'count'
    df_contribution_mcid_code_sum = df_contribution.groupby(groupby_cols, dropna=False).agg(agg_dict).reset_index()
    df_contribution_mcid_code_sum.rename(columns={'dos':'dos_count'}, inplace=True)
    #print(df_contribution_mcid_code_sum.columns)

    # mean contributions of the codes 
    cols = [col for col in df_contribution_mcid_code_sum.columns if any(x in col for x in ['mcid','type','codeIndex','contribCoef'])]
    groupby_cols = ['type','codeIndex']
    agg_cols = [col for col in cols if any(x in col for x in ['mcid', 'contribCoef'])]
    agg_dict = {col: 'sum' for col in agg_cols if 'mcid' not in col}
    agg_dict['mcid'] = 'count'
    df_contribution_code_summary = df_contribution_mcid_code_sum[cols].groupby(groupby_cols, dropna=False).agg(agg_dict).reset_index()
    df_contribution_code_summary.rename(columns={'mcid':'count'}, inplace=True)
    #print(df_contribution_code_summary.columns)
            
    return df_contribution,df_contribution_mcid_code_sum,df_contribution_code_summary

    
######Some help functions for reformatting the output from interpretation
#faltten the lists
def contribution_reformat(mcidList,y_obs, y_score, dosList, contrList, codeIndexList, type="diag",task = 'binary'):  
    """
    For sequential code, flatten the nested list to data frame
    Arguments:
        dosList: date of service of seq codes, list(list))
        contrList: contribution coef of seq codes, list(list(list)))
        codeIndexList: code index of seq codes, corresponding to contrList, list(list(list))) 
    outputs:
        df_contribution with columns ['mcid', 'y_obs', 'y_score', 'dos', 'contribCoef', 'codeIndex', 'type']
    """
    mcidList_all = []
    y_obsList_all = []
    y_scoreList_all = []
    dosList_all = []
    contribCoefList_all = []
    codeIndexList_all = []
    if task == 'multiclass':
        class_columns = {f'y_score_{i}': [] for i in range(len(y_score[0]))}
        num_classes = len(contrList)
        contrib_columns = {f'contribCoef_{i}': [] for i in range(num_classes)}
        #for classes,contrlists in enumerate(contrList):
            
        for i_patient, patient in enumerate(codeIndexList):
            for j_visit, visit in enumerate(patient):
                mcidList_all.extend([mcidList[i_patient]]*len(visit))
                if y_obs is not None :
                    y_obsList_all.extend([y_obs[i_patient]]*len(visit))
                y_scoreList_all.extend([y_score[i_patient]]*len(visit))
                dosList_all.extend([dosList[i_patient][j_visit]]*len(visit))
                #contribCoefList_all.extend(contrlists[i_patient][j_visit])
                codeIndexList_all.extend(codeIndexList[i_patient][j_visit])

                #creating y_score values into seperate columns for each classes
                for i, class_prob in enumerate(y_score[i_patient]):
                    class_columns[f'y_score_{i}'].extend([class_prob] * len(visit))
                #creating contribution-coef values into seperate columns for each classes
                for i_class in range(num_classes):
                    contrib_columns[f'contribCoef_{i_class}'].extend(contrList[i_class][i_patient][j_visit])

        #print(len(mcidList_all), len(y_obsList_all), len(y_scoreList_all), len(dosList_all), len(contribCoefList_all))
        if y_obs is None:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'dos': dosList_all,'codeIndex':codeIndexList_all, **class_columns,**contrib_columns})
        else:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_obs': y_obsList_all,'dos': dosList_all,'codeIndex':codeIndexList_all, **class_columns,**contrib_columns})
        df_contribution['type'] = type
        #df_contribution = df_contribution.drop_duplicates()
        #print(df_contribution.columns)
    else:
        for i_patient, patient in enumerate(contrList):
            for j_visit, visit in enumerate(patient):
                mcidList_all.extend([mcidList[i_patient]]*len(visit))
                if y_obs is not None :
                    y_obsList_all.extend([y_obs[i_patient]]*len(visit))
                y_scoreList_all.extend([y_score[i_patient]]*len(visit))
                dosList_all.extend([dosList[i_patient][j_visit]]*len(visit))
                contribCoefList_all.extend(contrList[i_patient][j_visit])
                codeIndexList_all.extend(codeIndexList[i_patient][j_visit])
        #print(len(mcidList_all), len(y_obsList_all), len(y_scoreList_all), len(dosList_all), len(contribCoefList_all))
        if y_obs is None:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_score': y_scoreList_all,
                                        'dos': dosList_all,'contribCoef':contribCoefList_all, 'codeIndex':codeIndexList_all})
        else:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_obs': y_obsList_all,'y_score': y_scoreList_all,
                                        'dos': dosList_all,'contribCoef':contribCoefList_all, 'codeIndex':codeIndexList_all})
        df_contribution['type'] = type
    return df_contribution

def contribution_seqPair_reformat(mcidList,y_obs, y_score,dosList,contrList,pairList, type="lab", task = 'binary'):  
    """
    For sequential pair, flatten the nested list to data frame
    Arguments:
        contrList: contribution coef of seq pair, list(list(list)))
        pairList: seq pair data,list(list(dict)))
    outputs:
        df_contribution with columns ['mcid', 'y_obs', 'y_score', 'dos', 'contribCoef', 'codeIndex', 'value', 'type']
    """
    mcidList_all = []
    y_obsList_all = []
    y_scoreList_all = []
    dosList_all = []
    contribCoefList_all = []
    codeIndexList_all = []
    pairValueList_all = []
    if task == 'multiclass':
        class_columns = {f'y_score_{i}': [] for i in range(len(y_score[0]))}
        num_classes = len(contrList)
        contrib_columns = {f'contribCoef_{i}': [] for i in range(num_classes)}
        #for classes,contrlists in enumerate(contrList):
        for i_patient, patient in enumerate(pairList):
            for j_visit, visit in enumerate(patient):
                mcidList_all.extend([mcidList[i_patient]]*len(visit))
                if y_obs is not None:
                    y_obsList_all.extend([y_obs[i_patient]]*len(visit))
                y_scoreList_all.extend([y_score[i_patient]]*len(visit))
                dosList_all.extend([dosList[i_patient][j_visit]]*len(visit))
                #contribCoefList_all.extend(contrlists[i_patient][j_visit])
                #codeIndexList_all.extend(codeIndexList[i_patient][j_visit])
                codeIndexList_all.extend(pairList[i_patient][j_visit].keys())
                pairValueList_all.extend(pairList[i_patient][j_visit].values())

                #creating y_score values into seperate columns for each classes
                for i, class_prob in enumerate(y_score[i_patient]):
                    class_columns[f'y_score_{i}'].extend([class_prob] * len(visit))
                #creating contribution-coef values into seperate columns for each classes
                for i_class in range(num_classes):
                    contrib_columns[f'contribCoef_{i_class}'].extend(contrList[i_class][i_patient][j_visit])

        #print(len(mcidList_all), len(y_obsList_all), len(y_scoreList_all), len(dosList_all), len(contribCoefList_all))
        if y_obs is None:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'dos': dosList_all,'codeIndex':codeIndexList_all, 'value':pairValueList_all,**class_columns,**contrib_columns})
        else:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_obs': y_obsList_all,'dos': dosList_all,'codeIndex':codeIndexList_all, 'value':pairValueList_all,**class_columns,**contrib_columns})
        df_contribution['type'] = type
    else:
        for i_patient, patient in enumerate(contrList):
            for j_visit, visit in enumerate(patient):
                mcidList_all.extend([mcidList[i_patient]]*len(visit))
                if y_obs is not None:
                    y_obsList_all.extend([y_obs[i_patient]]*len(visit))
                y_scoreList_all.extend([y_score[i_patient]]*len(visit))
                dosList_all.extend([dosList[i_patient][j_visit]]*len(visit))
                contribCoefList_all.extend(contrList[i_patient][j_visit])
                #codeIndexList_all.extend(codeIndexList[i_patient][j_visit])
                codeIndexList_all.extend(pairList[i_patient][j_visit].keys())
                pairValueList_all.extend(pairList[i_patient][j_visit].values())
        #print(len(mcidList_all), len(y_obsList_all), len(y_scoreList_all), len(dosList_all), len(contribCoefList_all), len(codeIndexList_all), len(pairValueList_all))
        if y_obs is None:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_score': y_scoreList_all,
                        'dos': dosList_all,'contribCoef':contribCoefList_all, 'codeIndex':codeIndexList_all, 'value':pairValueList_all})   
        else:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_obs': y_obsList_all,'y_score': y_scoreList_all,
                        'dos': dosList_all,'contribCoef':contribCoefList_all, 'codeIndex':codeIndexList_all, 'value':pairValueList_all})        
        df_contribution['type'] = type
    return df_contribution

def contribution_static_reformat(mcidList,y_obs,y_score,contrStat, xStat,cols=["age","gender"],task = 'binary'):
    """
    Reformat contrStat, xStat into dataframe of long format
    Arguments:
        contrStat: 2D tensor of contribution coef of static features for task = 'binary' or 'regression', if task = 'multiclass', list(2D tensor).
        xStat: 2D tensor of values of static features 
        cols: column names for static features 
    outputs:
        df_contribution with columns ['mcid', 'y_obs', 'y_score', 'contribCoef', 'code', 'value', 'type']
    """
    mcidList_all = []
    y_obsList_all = []
    y_scoreList_all = []
    contribCoefList_all = []
    valuesList_all = []
    typeList = []
    
    if task == 'multiclass':
        class_columns = {f'y_score_{i}': [] for i in range(len(y_score[0]))}
        num_classes = len(contrStat)
        contrib_columns = {f'contribCoef_{i}': [] for i in range(num_classes)}
        #for classes,contrstats in enumerate(contrStat):
        #for i_patient, patient in enumerate(contrstats):
        for i_patient in range(len(mcidList)):
            mcidList_all.extend([mcidList[i_patient]]*len(cols))
            if y_obs is not None:
                y_obsList_all.extend([y_obs[i_patient]]*len(cols))
            y_scoreList_all.extend([y_score[i_patient]]*len(cols))
            #contribCoef = torch.flatten(contrstats.t()).detach().cpu().numpy()
            #contribCoefList_all.extend(torch.flatten(contrstats[i_patient].t()).detach().cpu().numpy())
            #values = torch.flatten(xStat.t()).detach().cpu().numpy() 
            valuesList_all.extend(torch.flatten(xStat[i_patient].t()).detach().cpu().numpy()) 

            for i, class_prob in enumerate(y_score[i_patient]):
                class_columns[f'y_score_{i}'].extend([class_prob] * len(cols))

            #creating contribution-coef values into seperate columns for each classes
            for i_class in range(num_classes):
                contrib_columns[f'contribCoef_{i_class}'].extend(torch.flatten(contrStat[i_class][i_patient].t()).detach().cpu().numpy())

            for col in cols:
                typeList.extend([col])

        #print(len(mcidList_all),len(y_obsList_all),len(y_scoreList_all), len(contribCoefList_all),len(typeList),len(valuesList_all),len(class_columns['y_score_1']))
        if y_obs is None:
            df_contribution = pd.DataFrame({'mcid': mcidList_all, **class_columns,**contrib_columns,'codeIndex':typeList,'value':valuesList_all})
        else:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_obs': y_obsList_all, **class_columns,**contrib_columns,'codeIndex':typeList,'value':valuesList_all})
        df_contribution['type'] = 'static'
    else:
        n = len(contrStat)
        mcidList_all.extend(mcidList*len(cols))
        if y_obs is not None:
            y_obsList_all.extend(y_obs*len(cols))
        y_scoreList_all.extend(y_score*len(cols))
        contribCoef = torch.flatten(contrStat.t()).detach().cpu().numpy()
        values = torch.flatten(xStat.t()).detach().cpu().numpy() 
        for col in cols:
            typeList.extend([col] * n)
        print(len(mcidList),len(mcidList_all),contribCoef.shape, values.shape, len(typeList))
        if y_obs is None:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_score': y_scoreList_all,'contribCoef':contribCoef, 'codeIndex':typeList,'value':values})
        else:
            df_contribution = pd.DataFrame({'mcid': mcidList_all,'y_obs': y_obsList_all,'y_score': y_scoreList_all,'contribCoef':contribCoef, 'codeIndex':typeList,'value':values})
        df_contribution['type'] = 'static'
    return df_contribution


def idx2code(df, codedict, cols=['codeIndex', 'code', 'code_desc'], idx_offset = 0):
    """
    Merge the code index with the corresponding code and code_desc in codedict when the indecies are in one contigouous blocks
    codedict: with columns in cols 
    idx_offset: offset between index in df and index in codedict
    """
    df2 = codedict[cols]
    df2.columns = ['codeIndex', 'code', 'codeDesc']
    df['codeIndex'] = df['codeIndex'] + idx_offset
    df = pd.merge(df, df2, on='codeIndex', how = 'left')
    return df


def idx2code2(df, codedict, cols=['codeIndex', 'code', 'code_desc'], 
              idx_offset = 0, idx_end1=42684, idx_start2= 86061):
    """
    Merge the code index with the corresponding code and code_desc in codedict when the indecies are in two disconnected contigouous blocks. Foe example, diag codes in claim and EMR are in different blocks.
    codedict: with columns in cols 
    idx_offset: initial offset between index in df and index in codedict
    idx_end1: number of code indices for first contiguous block
    idx_start2: the starting index of second contigouous block of indices
    """
    df2 = codedict[cols]
    df2.columns = ['codeIndex', 'code', 'codeDesc']
    df['codeIndex'] = df['codeIndex'] + idx_offset
    cond = df['codeIndex'] >= idx_offset + idx_end1
    df.loc[cond, 'codeIndex'] = df.loc[cond, 'codeIndex']+idx_start2 - idx_offset - idx_end1   
    df = pd.merge(df, df2, on='codeIndex', how = 'left')
    return df   

def add_zro(x1):
    zro = torch.zeros((len(x1), 1), dtype=torch.float).to(str(x1.device)) 
    return torch.cat((x1, zro), dim = -1)
        
        
#Example: Model for 3 type of sequential codes (diag, proc, rx), 1 type of sequential pair (lab), 1 table of 2 static features
#model=CustomModel(num_seqCode = [num_codes1, num_codes2, num_codes3], num_seqPair=[num_codes_p1], num_static=2)
class CustomModel(nn.Module):
    """
    for binary and multi-class classification, and regression problem
    Inputs: 
        num_codes1, num_codes2, num_codes3: number of codes for sequential code1 to code3
        num_codes_p1, ...: number of codes for sequential pair
        num_static: num of static features in the table of static features
        e.g.: num_seqCode = [num_codes1, num_codes2, num_codes3], num_seqPair=[num_codes_p1], num_static=2
        num_classes: =1 for binary classification and regression, = #classes>1 for multi-class classification
        hiddenMethod: the method used to calculate the visit attention. It can be: None, 'gru', 'transformerEncoder'
        num_heads: number of heads, one head is one set of attentions (code, visit, and type attentions)
        p_dropout: probability of dropout
    """
    def __init__(self, num_classes=1, num_seqCode=[1,2], num_seqPair=[], num_static=2, embedding_dim=128, hiddenMethod ='gru', num_heads = 2, p_dropout=0.1): 
        super(CustomModel,self).__init__() 
        self.modls = nn.ModuleList(nn.ModuleList() for h in range(num_heads))
        #Sequential codes           
        for num in num_seqCode:
            embedding1 = nn.Embedding(num, embedding_dim, padding_idx=0)
            for h in range(num_heads):
                self.modls[h].append(SeqCodeUnit('feature_name'+str(num), embedding1, hiddenMethod, p_dropout))
        #sequential pairs
        for num in num_seqPair:
            embedding1 = nn.Embedding(num, embedding_dim, padding_idx=0)
            for h in range(num_heads):
                self.modls[h].append(SeqPairUnit('value_feature'+str(num), embedding1, hiddenMethod, p_dropout))
        #
        if num_static is not None and num_static > 0 :
            self.modls_static = StaticFeatureUnit("static_feature", num_static)
        
        self.feature_num = [0 if num_seqCode is None else len(num_seqCode),
                            0 if num_seqPair is None else len(num_seqPair),
                            0 if num_static is None or num_static == 0 else 1
                            ]
        #
        if num_seqPair: 
            embedding_dim = embedding_dim + 1
        self.attn_type = nn.ModuleList([Attention(embedding_dim) for h in range(num_heads)])
        if num_heads > 1:
            self.linear_mh= nn.Linear(num_heads, 1, bias = False)
        self.linear= nn.Linear(embedding_dim, num_classes)
        
        if self.feature_num[2] != 0 :
            self.linear_static = nn.Linear(num_static, num_classes, bias = False)

        #self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=p_dropout)
        
        self.num_classes = num_classes
        self.num_heads = num_heads

    def forward(self, *args, interpret = False):
        """
        Arguments:
            args can be:
            static_x: static features of shape (batch_size, num_static)
            x1, x2, x3, x4: for sequential features. Tensors of shape (batch_size, num_visits, num_notes, embedding_dim)
            masks1, masks2, masks3, masks4: padding masks of shape (batch_size, num_visits, num_notes) 
        """
        outputs1 = [[] for h in range(self.num_heads)]
        outputs2 = [[] for h in range(self.num_heads)]
        xss = [[] for h in range(self.num_heads)]
        masks = [[] for h in range(self.num_heads)]
        a_type = [[] for h in range(self.num_heads)]
        logits = [0 for h in range(self.num_heads)]
        prob_mh = [0 for h in range(self.num_heads)]
        if (self.feature_num[0] > 0):
            input_xs = [args[i] for i in range(0, 2 * self.feature_num[0], 2)]
            input_masks1 = [args[i] for i in range(1, 2 * self.feature_num[0], 2)]
            for h in range(self.num_heads):
                outputs1[h] = [module(x, mask, interpret) for module, x, mask in zip(self.modls[h][0 : self.feature_num[0]], input_xs, input_masks1)]
                xss[h] = [x[0] for x in outputs1[h]]
                masks[h] = [x[1][:,0,0] for x in outputs1[h]]

        if (self.feature_num[1] > 0):
            input_xs = [args[i + 2* self.feature_num[0] ] for i in range(0, 3 * self.feature_num[1], 3)]
            input_masks2 = [args[i  + 2* self.feature_num[0] ] for i in range(1, 3 * self.feature_num[1], 3)]
            input_vals = [args[i  + 2* self.feature_num[0] ] for i in range(2, 3 * self.feature_num[1], 3)]
            for h in range(self.num_heads):
                outputs2[h] = [ module(k, mask, val, interpret) for module, k, mask, val in zip(self.modls[h][self.feature_num[0] : (self.feature_num[0] + self.feature_num[1])], input_xs, input_masks2, input_vals)]
            
            #pad 0
            for h in range(self.num_heads):
                xss[h] = [ add_zro(x) for x in xss[h]]
                x2s = [x[0] for x in outputs2[h]]
                t2s = [x[1][:,0,0] for x in outputs2[h]]
                xss[h].extend(x2s)
                masks[h].extend(t2s)
         
        # TYPE level combination 
        #logits = 0
        x = [0 for h in range(self.num_heads)]
        for h in range(self.num_heads):
            #print('xxs[h]:',type(xss[h]), len(xss[h]), xss[h][0].shape)
            xx = torch.stack(xss[h], dim = -2)
            masks_type = torch.stack(masks[h], dim = -1)
            a_type[h] = self.attn_type[h](xx, masks_type)
            x[h] = attention_types_sum(a_type[h], xx, masks=masks_type)
            #print(x[h].shape)
        M =  torch.stack(x, dim = -1)
        if self.num_heads > 1:
            m = self.linear_mh(M).squeeze(dim = -1)
        else:
            m = M.squeeze(dim = -1)
        m=self.dropout(m)
        #print(M.shape, m.shape)
        logits = self.linear(m)
            
        # check static features , take as a whole
        z = None
        if self.feature_num[2] == 1 :
            x_static = args[ 2 * self.feature_num[0] + 3 * self.feature_num[1]  : ]
            #z = self.modls[0][ self.feature_num[0] + self.feature_num[1] ].to(x_static[0].device)(x_static[0], interpret)
            z = self.modls_static(x_static[0], interpret)
            logits = logits + self.linear_static(z)     
   
        if interpret:
            return logits, outputs1, outputs2, a_type, masks_type, z 
        else:   
            return logits
        

class SeqCodeUnit(nn.Module):
    """
    module to handle one type medical code processing from codeIdx to member representation
    
    """           
    def __init__(self, feature_name, embedding,method,p_dropout):    
        super(SeqCodeUnit, self).__init__() 
        self.name = feature_name
        self.embedding1 = embedding
        self.method = method
        embedding_dim = embedding.embedding_dim
        self.attn_note1 = Attention(embedding_dim)
        #self.gru1 = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        if method == 'gru':
            self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        if method == 'transformerEncoder':
            encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_visit1 = Attention(embedding_dim)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x1, masks1, interpret = False):
        """
        Arguments:
            x1: tensor of codeIdx of shape (batch_size, # visits, # codes)
            mask1: mask tensor of shape (batch_size, # visits, # codes)
        outputs:
            x: notes sequence of shape (batch_size, # visits, # codes, embedding_dim)
            masks: the padding masks of shape (batch_size, # visits, # codes) 
        """
        xx1 = self.embedding1(x1)
        #print(xx1.shape)
        a_note1 = self.attn_note1(xx1, masks1)
        x1 = attention_codes_sum(a_note1, xx1, masks=masks1)
        #print(x1.shape)
        if self.method is None:
            h1 = x1
        if self.method == 'gru':
            h1, _ = self.gru(x1)             #shape= (batch_size, #visits, embedding_dim). 
        if self.method == 'transformerEncoder':
            h1 = PositionalEncoding(x1)
            h1 = self.transformer_encoder(h1)  #shape= (batch_size, #visits, embedding_dim).
        #print('h1:', h1.shape)
        a_visit1 = self.attn_visit1(h1, masks1)                  #visit attention
        x = attention_visits_sum(a_visit1, x1, masks=masks1)
        x = self.dropout(x)

        if interpret:
            return x, masks1, xx1, a_note1, a_visit1
        else:
            return x, masks1
        
 
class SeqPairUnit(nn.Module):
    """
    moudle to handle k:v pair values
    """
    def __init__(self, feature_name, embedding,method,p_dropout): 
        super(SeqPairUnit, self).__init__() 
        self.feature_name = feature_name
        self.method = method
        #sequential pairs
        self.embedding_p1 = embedding
        embedding_dim = embedding.embedding_dim 
        self.attn_note_p1 = Attention(embedding_dim+1)
        #self.gru_p1 = nn.GRU(embedding_dim+1, embedding_dim+1, batch_first=True)
        if method == 'gru':
            self.gru_p1 = nn.GRU(embedding_dim+1, embedding_dim+1, batch_first=True)
        if method == 'transformerEncoder':
            encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim+1, nhead=1, batch_first=True)   #default to 2 heads
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)                  #default to 2 layers of multi-heads
        self.attn_visit_p1 = Attention(embedding_dim+1)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x_key_p1, mask_key_p1, x_value_p1, interpret = False):
        """
        Arguments:
        """
        #sequential pairs
        xx_p1 = self.embedding_p1(x_key_p1) #(batch_size, #visits, #codes, embedding_dim)
        xx_p1 = torch.cat((xx_p1, x_value_p1.unsqueeze(-1)), dim = -1)
        a_note_p1 = self.attn_note_p1(xx_p1, mask_key_p1)
        x_p1 = attention_codes_sum(a_note_p1, xx_p1, masks=mask_key_p1)
        if self.method is None:
            h_p1 = x_p1
        if self.method == 'gru':
            h_p1, _ = self.gru_p1(x_p1)             #shape= (batch_size, #visits, embedding_dim). 
        if self.method == 'transformerEncoder':
            h_p1 = PositionalEncoding(x_p1)
            h_p1 = self.transformer_encoder(h_p1)  #shape= (batch_size, #visits, embedding_dim).
        a_visit_p1 = self.attn_visit_p1(h_p1, mask_key_p1)                  #visit attention
        x_p1 = attention_visits_sum(a_visit_p1, x_p1, masks=mask_key_p1)
        #print('x_p1', x_p1.shape, x_p1.device)
        x_p1 = self.dropout(x_p1)
        
        #pad 0
        if interpret:
            return x_p1, mask_key_p1, xx_p1, a_note_p1, a_visit_p1
        else:
            return x_p1, mask_key_p1
        
class StaticFeatureUnit(nn.Module):
    """
    modeul to handle static values 
    """
    def __init__(self, feature_name, num_static): 
        super(StaticFeatureUnit, self).__init__() 
        self.feature_name = feature_name
        #static features
        self.bn = nn.BatchNorm1d(num_static)

    def forward(self, x_static, interpret = False):
        """
        Arguments:
            static_x: static features of shape (batch_size, num_static)
        """
        #static features
        z = self.bn(x_static)
        
        return z 
     