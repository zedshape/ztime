from .utils import getRelation, Database
import numpy as np
import pandas as pd
from collections import defaultdict

class ZTime:
    def __init__(self, step, forgettable=True):
        self.FL = defaultdict(lambda: defaultdict(int))
        self.comparisoncount = 0
        self.totalfrequency = 0
        self.forgettable = forgettable
        self.step = step
        self.saveloc = False
        self.isTest = False
        self.initialSupports = None

    def pruneTestEventLabels(self):
        for seq in self.database.sequences:
            prunedSequences = []
            for event in seq.sequences:
                print(event, event.label)
                if event.label in self.initialSupports:
                    prunedSequences.append(event)
            seq.sequences = prunedSequences
        return

    def createZTableWithLocation(self):
        self.loc = defaultdict(lambda: defaultdict(set))

        for sid in range(len(self.database)):
            S = self.database[sid]
            
            for s1_idx in range(len(S)):
            
                max_range = np.min([len(S), s1_idx + self.step])
                tmp_pair = set()
                for s2_idx in range(s1_idx, max_range):
                    s1 = S[s1_idx]
                    s2 = S[s2_idx]
                    
                    if self.step == 0:
                        break

                    R2 = getRelation(s1, s2)
                    
                    if R2 != None:
                        E2 = (s1[2], s2[2], R2)
                        
                        if ((self.isTest == False) or ((self.isTest == True) and (E2 in self.trainArrangementsSet))):
                            
                            self.loc[E2][sid].add((s1, s2))
                            if E2 not in tmp_pair:
                                self.FL[E2][sid] += 1
                                tmp_pair.add(E2)
                                
    def createZTable(self):
        for sid in range(len(self.database)):
            S = self.database[sid]

            for s1_idx in range(len(S)):

                max_range = np.min([len(S), s1_idx + self.step])
                tmp_pair = set()
                for s2_idx in range(s1_idx, max_range):
                
                    s1 = S[s1_idx]
                    s2 = S[s2_idx]

                    if self.step == 0:
                       break

                    R2 = getRelation(s1, s2)
                    
                    if R2 != None:

                        E2 = (s1[2], s2[2], R2)
                        if ((self.isTest == False) or ((self.isTest == True) and (E2 in self.trainArrangementsSet))):                   
                            if E2 not in tmp_pair:
                                self.FL[E2][sid] += 1
                                tmp_pair.add(E2)

    def chooseTopKFeatures(self, k, op = "vertical"):
        data_pd = pd.DataFrame.from_dict(self.FL).fillna(0)
        data = data_pd.to_numpy()
        summation = np.zeros(0)
        if k <= data.shape[1]:
            if op == "vertical":
                summation = np.count_nonzero(data, axis=0)
            elif op == "horizontal":
                summation = np.sum(data, axis=0)
            elif op == "relat_sup":
                ratios = []
                for lab in np.unique(self.labels):
                    indices = self.labels == lab
                    set1 = data[indices]
                    count1 = np.count_nonzero(set1, axis=0)
                    riskratio = (count1 / len(set1))
                    ratios.append(riskratio)
                ratios = np.array(ratios)
                summation = np.average(ratios, axis=0)
            elif op in ["disprop_max", "disprop_avg"]:
                ratios = []
                for lab in np.unique(self.labels):
                    indices = self.labels == lab
                    set1 = data[indices]
                    set2 = data[~indices]
                    count1 = np.count_nonzero(set1, axis=0)
                    count2 = np.count_nonzero(set2, axis=0)
                    riskratio = ((count1+1) / (count2+1)) / (len(set1) / len(set2))
                    ratios.append(riskratio)
                ratios = np.array(ratios)

                if op == "disprop_max":
                    summation = np.max(ratios, axis=0)
                if op == "disprop_avg":
                    summation = np.average(np.abs(ratios - 1), axis=0)
            
            
            indices = np.argpartition(summation, -k)[-k:]
            selectedColumns = data_pd.iloc[:, indices].columns.values
            self.FL = {k: self.FL[k] for k in selectedColumns}

    def resetParams(self):
        self.FL = defaultdict(lambda: defaultdict(int))
        self.comparisoncount = 0
        self.totalfrequency = 0
        self.database = None
        
    def test(self, database, saveloc = False):
        self.isTest = True
        self.saveloc = saveloc
        self.resetParams()
        self.database = database
        for key in self.trainArrangementsSet:
            self.FL[key] = defaultdict(int)
        self.trainArrangementsSet = set(self.trainArrangementsSet)
        self.run()
        self.database = None
        return self.FL

    def train(self, database, labels):
        self.database: Database = database
        self.labels = labels
        self.run()
        self.trainArrangementsSet = list(self.FL.keys())
        return self.FL

    def run(self):
        if self.saveloc == False:
            self.createZTable()
        else:
            self.createZTableWithLocation()