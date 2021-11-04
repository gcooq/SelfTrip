#!/usr/bin/python3
# coding=utf-8
import tensorflow as tf


# ----------------------------------- Hidden_init ---------------------------------------
class Hidden_init(tf.keras.Model):
    def __init__(self, hidden_size):
        super(Hidden_init, self).__init__()
        self.hidden_size = hidden_size
        self.fc = tf.keras.layers.Dense(self.hidden_size, name='hidden_init')

    def call(self, input_que):
        hidden_state = tf.nn.relu(self.fc(input_que))

        return hidden_state

    def reset_variable(self):
        self.fc = tf.keras.layers.Dense(self.hidden_size, name='hidden_init')


# ----------------------------------- QueryModel ---------------------------------------
class QueryModel(tf.keras.Model):
    def __init__(self, poi_embedding, K_dim):
        super(QueryModel, self).__init__()
        self.K_dim = K_dim
        self.poi_dim = poi_embedding.shape[1] + 24
        self.w = tf.Variable(tf.random.normal([2 * self.poi_dim, self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.b = tf.Variable(tf.random.normal([self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.K = tf.Variable(
            tf.random.normal([self.poi_dim, self.poi_dim, self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.poi_embedding = poi_embedding
        self.time_embdding = tf.one_hot([i for i in range(24)], 24, dtype='float64')

    def call(self, query_batch, training=None, mask=None):
        start_poi = tf.nn.embedding_lookup(self.poi_embedding, query_batch[:, 0])
        end_poi = tf.nn.embedding_lookup(self.poi_embedding, query_batch[:, 2])
        start_time = tf.nn.embedding_lookup(self.time_embdding, query_batch[:, 1])
        end_time = tf.nn.embedding_lookup(self.time_embdding, query_batch[:, 3])

        start_poi = tf.concat([start_poi, start_time], 1)
        end_poi = tf.concat([end_poi, end_time], 1)
        X = tf.concat([start_poi, end_poi], 1)

        out = tf.matmul(X, self.w) + self.b + \
              tf.matmul(end_poi, tf.reduce_sum(tf.matmul(start_poi, self.K), 1))

        return tf.nn.leaky_relu(out)

    def reset_variable(self):
        self.w = tf.Variable(tf.random.normal([2 * self.poi_dim, self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.b = tf.Variable(tf.random.normal([self.K_dim], mean=0, stddev=1, dtype='float64'))
        self.K = tf.Variable(
            tf.random.normal([self.poi_dim, self.poi_dim, self.K_dim], mean=0, stddev=1, dtype='float64'))


# ----------------------------------- Decoder ---------------------------------------
class Decoder(tf.keras.Model):
    def __init__(self, poi_embedding, poi_size, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.poi_size = poi_size
        self.dr = 0.5
        self.embedding = poi_embedding
        self.gru = tf.keras.layers.GRUCell(self.dec_units, dropout=self.dr)
        self.fc = tf.keras.layers.Dense(poi_size)
        self.fc2 = tf.keras.layers.Dense(poi_size)
        self.fc_0 = tf.keras.layers.Dense(tf.shape(self.embedding)[0])

    def call(self, x, query, dec_hidden):
        x1 = tf.nn.embedding_lookup(self.embedding, x)
        query = self.fc_0(query)
        x1 = tf.concat([x1, query], axis=1)

        output, state = self.gru(x1, dec_hidden)
        output2 = output

        x1 = self.fc(output)
        x2 = self.fc2(output2)

        return x1, x2, state

    def pre_train(self, x, query, dec_hidden):
        query = self.fc_0(query)
        x = tf.concat([x, query], axis=1)
        output, state = self.gru(x, dec_hidden)

        return output, state

    def reset_variable(self):
        self.gru = tf.keras.layers.GRUCell(self.dec_units)
        self.fc = tf.keras.layers.Dense(self.poi_size)
        self.fc2 = tf.keras.layers.Dense(self.poi_size)
        self.fc_0 = tf.keras.layers.Dense(tf.shape(self.embedding)[0])

    def set_dropout(self):
        self.dr = 1.0
