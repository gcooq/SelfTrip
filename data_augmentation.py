#!/usr/bin/python3
# coding=utf-8

import numpy as np
import pandas as pd
import random
import copy


class DataAugmentation():

    def __init__(self, city):
        self.city = city
        self.poi_embedding = pd.read_csv('./self-embedding/'+self.city+'_poi_weight.csv')
        self.poi_size = self.poi_embedding.shape[0]
        self.poi_em_size = self.poi_embedding.shape[1]

    def original(self):
        return self.poi_embedding

    def dropout(self,pro=0.5):
        self.drop_em = pd.DataFrame.copy(self.poi_embedding)
        drop_num = int(self.poi_em_size * pro)
        for i in range(self.poi_size):
            drop_list = np.random.randint(0,self.poi_em_size-1, drop_num)
            for j in drop_list:
                self.drop_em.iloc[i, j] = 0

        return self.drop_em

    def token_shuffing(self, traj):
        self.token_shuffing_em = pd.DataFrame.copy(self.poi_embedding)
        row_list = copy.deepcopy(traj)
        random.shuffle(traj)
        poi_dict = dict(zip(row_list,traj))
        for i in self.poi_embedding.index:
            if(i in row_list):
                self.token_shuffing_em.iloc[i] = self.poi_embedding.iloc[poi_dict[i]]
            self.token_shuffing_em.iloc[i] = self.poi_embedding.iloc[i]

        return self.token_shuffing_em

    def token_cutoff(self, pro=0.2):
        self.token_cutoff_em = pd.DataFrame.copy(self.poi_embedding)
        drop_num = int(self.poi_size * pro)
        drop_list = np.random.randint(0, self.poi_size - 1, drop_num)
        for i in drop_list:
            self.token_cutoff_em.iloc[i] = 0

        return self.token_cutoff_em

    def feature_cutoff(self, pro=0.2):
        self.feature_cutoff_em = pd.DataFrame.copy(self.poi_embedding)
        drop_num = int(self.poi_size * pro)
        drop_list = np.random.randint(0, self.poi_size - 1, drop_num)
        for i in drop_list:
            self.feature_cutoff_em[str(i)] = 0

        return self.feature_cutoff_em

