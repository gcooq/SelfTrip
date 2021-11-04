#!/usr/bin/python3
# coding=utf-8
import tensorflow as tf
import time
from input_data import Load_data
from model import QueryModel
from model import Decoder
from model import Hidden_init
from metric import *


city = 'Osak'
# city = 'Glas'
# city = 'Edin'
# city = 'Toro'
tf.keras.backend.set_floatx('float64')# float desgin

pretrain_batch_size = 32
batch_size = 32
k = 256
dec_units = 256

data = Load_data(city)
poi_embedding, poi_size = data.self_embedding()
data_size = data.data_size()

query = QueryModel(poi_embedding, k)
decoder = Decoder(poi_embedding, poi_size, dec_units)
h_state = Hidden_init(dec_units)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred, real2, pred2):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    loss = loss_object(real2, pred2)

    return tf.reduce_mean(loss_) + 1 * tf.reduce_mean(loss)


def pre_loss_function(sim):
    real = np.eye(pretrain_batch_size)
    loss = tf.keras.losses.categorical_crossentropy(real, sim)

    return tf.reduce_mean(loss)


# --------------------------- pre-train_step ------------------------------
def pre_train_step(pre_que, sample1, sample2, lr=0.0005):
    pre_loss = 0
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    with tf.GradientTape() as tape:
        query_out = query(pre_que)
        dec_input1 = sample1[:, 0]
        dec_input2 = sample2[:, 0]
        dec_hidden1 = h_state(query_out)
        dec_hidden2 = h_state(query_out)

        for t in range(1, sample1.shape[1]):
            output1, dec_hidden1 = decoder.pre_train(dec_input1, query_out, dec_hidden1)
            output2, dec_hidden2 = decoder.pre_train(dec_input2, query_out, dec_hidden2)
            dec_input1 = sample1[:, t]
            dec_input2 = sample2[:, t]
        sim = tf.matmul(output1, tf.transpose(output2))
        sim = tf.math.softmax(sim)
        pre_loss += pre_loss_function(sim)

    batch_loss = pre_loss
    variables = decoder.trainable_variables + query.trainable_variables + h_state.trainable_variables
    gradients = tape.gradient(pre_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


exps = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    0.1,
                                                    decay_steps=5,
                                                    decay_rate=0.9,
                                                    staircase=False)

# --------------------------- train_step ---------------------------------
def train_step(que, traj,lr = 0.1):
    loss = 0
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    with tf.GradientTape() as tape:
        query_out = query(que)
        dec_input = traj[:, 0]
        dec_hidden=h_state(query_out)

        for t in range(1, traj.shape[1]):
            predictions, predictions2, dec_hidden = decoder(dec_input, query_out, dec_hidden)
            loss += loss_function(traj[:, t], predictions, que[:, 2], predictions2)
            dec_input = tf.argmax(tf.nn.softmax(predictions), 1)

    batch_loss = loss
    variables = query.trainable_variables + decoder.trainable_variables+h_state.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


# ------------------------------- evaluate ------------------------------
def evaluate(que, traj):
    predict_traj = []
    realnum_poi = 0
    query_out = query(que)
    dec_input = traj[:, 0]

    for poi in tf.squeeze(traj):
        if (poi == 0):
            break
        realnum_poi += 1
    realnum_poi = realnum_poi - 2

    start_poi = traj[:, 0]
    start_poi = tf.cast(start_poi,dtype=tf.int32)
    end_poi = traj[:, realnum_poi + 1]
    predict_traj.append(start_poi)
    dec_hidden = h_state(query_out)
    table = np.ones([poi_size],dtype=np.float64)

    table[start_poi.numpy()] = 0.
    table[end_poi.numpy()] = 0.
    table[0] = 0.

    for t in range(realnum_poi):
        decoder.set_dropout()
        predictions, _, dec_hidden = decoder(dec_input, query_out,dec_hidden)
        mask = tf.expand_dims(table,axis=0)
        dec_input = tf.argmax(tf.nn.softmax(predictions * mask), 1)
        predict_traj.append(dec_input)

    predict_traj.append(end_poi)
    real_traj = tf.squeeze(traj)[0:realnum_poi + 1].numpy()
    real_traj = np.append(real_traj, end_poi)
    predict_traj = [i.numpy().tolist() for i in predict_traj]
    predict_traj = [i[0] for i in predict_traj]

    batch_f1 = calc_F1(real_traj, predict_traj)
    batch_pairs_f1 = calc_pairsF1(real_traj, predict_traj)

    return batch_f1, batch_pairs_f1,real_traj, predict_traj


if __name__ == '__main__':
    total_test_f1 = []
    total_test_pairs_f1 = []

    max_f1 = []
    max_pf1 = []
    for index in range(data_size):
        decoder.reset_variable()
        query.reset_variable()
        h_state.reset_variable()

        pre_dataset_train, _, pre_steps_train, _ = data.load_dataset_one(index, pretrain_batch_size)
        # -------------------------- pre-train ----------------------------
        PRE_EPOCHES = 20
        for epoch in range(PRE_EPOCHES):
            start = time.time()
            pre_loss = 0
            for (batch, (que, traj)) in enumerate(pre_dataset_train.take(pre_steps_train)):
                total_batch_loss = 0
                pre_que, sample1, sample2 = data.load_pretrain_dataset(que, traj)
                pre_batch_loss = pre_train_step(pre_que, sample1, sample2)
                pre_loss += pre_batch_loss
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, pre_loss / pre_steps_train))
            print('Time taken for 1 epoch {} sec'.format(time.time() - start))

        # ------------------------------ train -----------------------------------
        dataset_train, dataset_val, steps_train, steps_val = data.load_dataset_one(index, batch_size)
        EPOCHS = 20
        start1 = time.time()
        test_f1 = []
        test_pairs_f1 = []
        res = {}
        for epoch in range(EPOCHS):
            start = time.time()
            total_loss = 0
            lr = 0.1
            for (batch, (que, traj)) in enumerate(dataset_train.take(steps_train)):
                batch_loss = train_step(que, traj, lr)
                total_loss += batch_loss

            total_f1 = 0
            total_pairs_f1 = 0

            for (batch, (que, traj)) in enumerate(dataset_val.take(steps_val)):
                for i in range(1):
                    traj1=traj[i].numpy()
                    indexs = np.where(traj1 == 0)
                    traj1 = np.delete(traj1,indexs)
                    traj1 = tf.convert_to_tensor(traj1)
                    que1 = tf.expand_dims(que[i], 0)
                    traj1 = tf.expand_dims(traj1, 0)

                    batch_f1, batch_ps_f1, real_traj, predict_traj = evaluate(que1, traj1)
                    total_f1 += batch_f1
                    total_pairs_f1 += batch_ps_f1

            test_f1.append(total_f1)
            test_pairs_f1.append(total_pairs_f1)
            res.setdefault(total_f1,[]).append(total_pairs_f1)

        total_test_f1.append(max(test_f1))
        total_test_pairs_f1.append(max(test_pairs_f1))
        keys=res.keys()
        keys=sorted(keys)

        max_f1.append(keys[-1])
        max_pf1.append(max(res[keys[-1]]))
        print('index f1-score: ', str(max(test_f1)) + '  index pairs f1-score: ', str(max(test_pairs_f1)))

        print('\n--' * 2 + 'Time take {} sec'.format(time.time() - start1))
        print('---- finish' + ' index = ' + str(index) + '-' * 5 + '\n')

    print('total f1-score: ', np.mean(total_test_f1))
    print('total pairs f1-score: ', np.mean(total_test_pairs_f1))

    print('max f1-score: ', np.mean(max_f1))
    print('max pairs f1-score: ', np.mean(max_pf1))


