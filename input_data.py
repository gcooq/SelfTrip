#!/usr/bin/python3
# coding=utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from data_augmentation import DataAugmentation


class Load_data():
    def __init__(self,city):
        self.city = city
        self.data_aug = DataAugmentation(city)

    def data_size(self):
        query_list = []
        query_data = pd.read_csv('./train_data/'+self.city+'-query.csv')
        for i in query_data.index:
            query_list.append(query_data.loc[i].tolist())

        return len(query_list)

    def random_embedding(self):
        poi_list = []
        poi_data = pd.read_csv('./self-embedding/'+self.city+'_poi_weight.csv')
        em_size = poi_data.shape[1]
        for i in poi_data.index:
            poi_list.append(np.random.randn(em_size).tolist())
        poi_size = len(poi_list)
        embedding = tf.constant(poi_list,dtype='float64')

        return embedding, poi_size

    def self_embedding(self):
        poi_list = []
        poi_data = pd.read_csv('./self-embedding/'+self.city+'_poi_weight.csv')
        for i in poi_data.index:
            poi_list.append(poi_data.loc[i].tolist())
        poi_size = len(poi_list)
        embedding = tf.constant(poi_list,dtype='float64')

        return embedding, poi_size

    def load_dataset(self,BATCH_SIZE):
        query_list = []
        trajs_list = []
        query_data = pd.read_csv('./train_data/'+self.city+'-query.csv')
        trajs_data = open('./train_data/'+self.city+'-trajs.dat','r')
        for i in query_data.index:
            query_list.append(query_data.loc[i].tolist())
        for line in trajs_data.readlines():
            tlist = [eval(i) for i in line.split()]
            trajs_list.append(tlist)
        print('total number:',len(query_list),len(trajs_list))
        trajs_list = tf.keras.preprocessing.sequence.pad_sequences(trajs_list, padding='post')
        query_train, query_val, trajs_train, trajs_val = train_test_split(query_list,trajs_list,test_size=0.2)

        print('train_set:',len(query_train),len(trajs_train))
        print('test_set:', len(query_val), len(trajs_val))

        dt_train = tf.data.Dataset.from_tensor_slices((query_train, trajs_train)).shuffle(len(query_train))
        dt_train = dt_train.batch(BATCH_SIZE, drop_remainder=True)

        dt_val = tf.data.Dataset.from_tensor_slices((query_val, trajs_val)).shuffle(len(query_val))
        dt_val = dt_val.batch(BATCH_SIZE, drop_remainder=True)

        return dt_train, dt_val, int(len(query_train)/BATCH_SIZE), int(len(query_val)/BATCH_SIZE)

    def load_pretrain_dataset(self, que, traj):

        pre_que = que
        r1 = random.randint(1, 4)
        r2 = random.randint(1, 4)

        sample1 = self.gen_random_sample(r1, traj)
        sample2 = self.gen_random_sample(r2, traj)

        return pre_que, sample1, sample2

    def gen_random_sample(self, rand_num, traj):

        if (rand_num == 0):
            return tf.nn.embedding_lookup(self.data_aug.original(), traj)

        if (rand_num == 1):
            return tf.nn.embedding_lookup(self.data_aug.token_cutoff(), traj)

        if (rand_num == 2):
            trajs = traj.numpy()
            sample = []
            for traj in trajs:
                traj_aug = tf.nn.embedding_lookup(self.data_aug.token_shuffing(traj), traj)
                sample.append(traj_aug.numpy())
            return tf.constant(sample, dtype='double')

        if (rand_num == 3):
            return tf.nn.embedding_lookup(self.data_aug.feature_cutoff(), traj)

        if(rand_num == 4):
            return tf.nn.embedding_lookup(self.data_aug.dropout(), traj)

    def load_dataset_one(self, index, BATCH_SIZE):

        query_list = []
        trajs_list = []
        query_data = pd.read_csv('./train_data/'+self.city+'-query.csv')
        trajs_data = open('./train_data/'+self.city+'-trajs.dat','r')
        for i in query_data.index:
            query_list.append(query_data.loc[i].tolist())
        for line in trajs_data.readlines():
            tlist = [eval(i) for i in line.split()]
            trajs_list.append(tlist)
        print('total number：',len(query_list),len(trajs_list))
        trajs_list = tf.keras.preprocessing.sequence.pad_sequences(trajs_list, padding='post')
        trajs_list = trajs_list.tolist()

        query_val = []
        trajs_val = []
        query_val.append(query_list.pop(index))
        trajs_val.append(trajs_list.pop(index))

        query_train = query_list
        trajs_train = trajs_list

        print('train set:',len(query_train),len(trajs_train))
        print('test set:', len(query_val), len(trajs_val))

        dt_train = tf.data.Dataset.from_tensor_slices((query_train, trajs_train)).shuffle(len(query_train))
        dt_train = dt_train.batch(BATCH_SIZE, drop_remainder=True)
        # print(dt_train)

        dt_val = tf.data.Dataset.from_tensor_slices((query_val, trajs_val)).shuffle(len(query_val))
        dt_val = dt_val.batch(1, drop_remainder=True)

        return dt_train, dt_val, int(len(query_train)/BATCH_SIZE), 1

    def load_dataset_train(self, BATCH_SIZE):
        query_train = []
        trajs_train = []
        query_data = pd.read_csv('./train_data/' + self.city + '-query-train.csv')
        trajs_data = open('./train_data/' + self.city + '-trajs-train.dat', 'r')
        for i in query_data.index:
            query_train.append(query_data.loc[i].tolist())
        for line in trajs_data.readlines():
            tlist = [eval(i) for i in line.split()]
            trajs_train.append(tlist)
        print('训练集总量：', len(query_train), len(trajs_train))
        trajs_train = tf.keras.preprocessing.sequence.pad_sequences(trajs_train, padding='post')

        dt_train = tf.data.Dataset.from_tensor_slices((query_train, trajs_train)).shuffle(len(query_train))
        dt_train = dt_train.batch(BATCH_SIZE, drop_remainder=True)

        return dt_train, int(len(query_train) / BATCH_SIZE)

    def load_dataset_test(self, BATCH_SIZE):
        query_test = []
        trajs_test = []
        query_data = pd.read_csv('./train_data/' + self.city + '-query-test.csv')
        trajs_data = open('./train_data/' + self.city + '-trajs-test.dat', 'r')
        for i in query_data.index:
            query_test.append(query_data.loc[i].tolist())
        for line in trajs_data.readlines():
            tlist = [eval(i) for i in line.split()]
            trajs_test.append(tlist)
        print('train set number:', len(query_test), len(trajs_test))
        trajs_test = tf.keras.preprocessing.sequence.pad_sequences(trajs_test, padding='post')

        dt_test = tf.data.Dataset.from_tensor_slices((query_test, trajs_test)).shuffle(len(query_test))
        dt_test = dt_test.batch(BATCH_SIZE, drop_remainder=True)

        return dt_test, int(len(query_test) / BATCH_SIZE)

