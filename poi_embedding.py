#!/usr/bin/python3
# coding=utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
poi_embedding_size = 250

class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    super(Word2Vec, self).__init__()
    self.query_embedding = Embedding(vocab_size,
                                     embedding_dim,
                                     input_length=1,
                                     name="query_em_layer")
    self.poi_embedding = Embedding(vocab_size,
                                   embedding_dim,
                                   input_length=num_ns+1,
                                   name='poi_em_layer')
    self.dots = Dot(axes=(3, 1))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    word_emb0 = self.query_embedding(target[:, 0])
    word_emb1 = self.query_embedding(target[:, 1])
    word_emb = (word_emb0 + word_emb1)/2
    context_emb = self.poi_embedding(context)
    dots = self.dots([context_emb, word_emb])

    return self.flatten(dots)


class PoiEmbedding():

    def __init__(self, dis_matric, train_data, poi_num, num_ns=3, walk_num=10, walk_length=6):

        self.dis_matric = dis_matric
        self.walk_num = walk_num
        self.walk_length = walk_length
        self.train_data = train_data
        self.poi_num = poi_num
        self.num_ns = num_ns
        self.sentences = []
        self.positive_data = []

        self.dataset = None

    def gen_sentences(self, start, end):
        for i in range(self.walk_num):
            sentence = [start, ]
            poi_now = start
            for j in range(self.walk_length):
                if np.sum(self.poi_graph.loc[poi_now]) != 1:
                    break
                poi_next = np.random.choice(self.poi_graph.columns, p=self.poi_graph.loc[poi_now])

                if poi_next == end:
                    sentence.append(poi_next)
                    break
                sentence.append(poi_next)
                poi_now = poi_next

            if (len(sentence) >= 3 and sentence[-1] == end):
                self.sentences.append(sentence)


    def gen_train(self):
        self.poi_graph = pd.DataFrame(np.zeros(self.poi_num**2).reshape(self.poi_num,self.poi_num),
                                      index=range(0, self.poi_num),
                                      columns=range(0, self.poi_num))
        for traj in self.train_data:
            for i in range(len(traj)-1):
                if(traj[i+1] == 0):
                    break
                self.poi_graph.loc[traj[i],traj[i+1]] += 1
        self.poi_graph = self.poi_graph + self.dis_matric

        for i in self.poi_graph.index:
            if np.sum(self.poi_graph.loc[i]) == 0:
                continue
            self.poi_graph.loc[i] = self.poi_graph.loc[i] / np.sum(self.poi_graph.loc[i])
        print(self.poi_graph)

        for start in self.poi_graph.index:
            for end in self.poi_graph.columns:
                self.gen_sentences(start, end)

        print('deepwalk data ', len(self.sentences))

        for sentence in self.sentences:
            target = (sentence[0],sentence[-1])
            for poi in sentence[1:-1]:
                self.positive_data.append((target,poi))

        print('正样本：',len(self.positive_data))

        targets, contexts, labels = [], [], []
        for target_pois, context_poi in self.positive_data:
            context_class = tf.expand_dims(
                tf.constant([context_poi], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=self.num_ns,
                unique=True,
                range_max=self.poi_num,
                seed=40,
                name="nagetive_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * self.num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_pois)
            contexts.append(context)
            labels.append(label)
        print(len(targets),len(contexts),len(labels))

        BATCH_SIZE = 4
        BUFFER_SIZE = 10000
        self.dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        self.dataset = self.dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    def train(self, city, em_size):
        embedding_dim = em_size   # poi嵌入维度
        word2vec = Word2Vec(self.poi_num, embedding_dim, self.num_ns)
        word2vec.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        word2vec.fit(self.dataset, epochs=20, callbacks=[tensorboard_callback])

        que_weights = word2vec.get_layer('query_em_layer').get_weights()[0]
        poi_weights = word2vec.get_layer('poi_em_layer').get_weights()[0]
        que_weights = pd.DataFrame(que_weights)
        print(que_weights.shape)
        poi_weights = pd.DataFrame(poi_weights)
        print(poi_weights.shape)
        # que_weights.to_csv('./self-embedding/'+city+'_que_weight.csv',index=False)
        # poi_weights.to_csv('./self-embedding/'+city+'_poi_weight.csv',index=False)


if __name__ == '__main__':

    city = 'Osak'
    # city = 'Glas'
    # city = 'Edin'
    # city = 'Toro'
    trajs_data = open('./train_data/'+city+'-trajs.dat','r')
    trajs_list = []
    poi_dis_matric = pd.read_csv('./dis_matric/' + city + '_dis_matric.csv')

    for line in trajs_data.readlines():
        tlist = [eval(i) for i in line.split()]
        trajs_list.append(tlist)
    print('total number', len(trajs_list))

    poi_size = poi_dis_matric.shape[0]  # poi个数
    print('poi number', poi_size)

    poi_dis_matric.index = [i for i in range(0, poi_size)]
    poi_dis_matric.columns = [i for i in range(0, poi_size)]

    self_embedding = PoiEmbedding(poi_dis_matric, trajs_list, poi_size)
    self_embedding.gen_train()
    self_embedding.train(city, poi_embedding_size)

