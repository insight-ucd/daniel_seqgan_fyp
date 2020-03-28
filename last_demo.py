from __future__ import print_function

import codecs

__doc__ = """Char-based Seq-GAN on data from a book."""

import model
import train
import lastFM

import os.path
import numpy as np
import tensorflow as tf
import random
import time
import subprocess
import gzip

EMB_DIM = 20
HIDDEN_DIM = 25
SEQ_LENGTH = 10
START_TOKEN = 0

EPOCH_ITER = 1000
CURRICULUM_RATE = 0.02  # how quickly to move from supervised training to unsupervised
TRAIN_ITER = 100000  # generator/discriminator alternating
D_STEPS = 3  # how many times to train the discriminator per generator step
SEED = 88




def tokenize(s):
    return [c for c in ' '.join(s.split())]


def get_data():
    token_stream, all_tracks = lastFM.get_sessions(100000)
    print("TS: ",type(token_stream), len(token_stream), type(token_stream.values.tolist()))
    return token_stream, all_tracks


class BookGRU(model.GRU):

    def d_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate


def get_trainable_model(num_emb):
    return BookGRU(
        num_emb, EMB_DIM, HIDDEN_DIM,
        SEQ_LENGTH, START_TOKEN)


def get_random_sequence(token_stream):
    """Returns random subsequence."""


    while True:
        row_idx = random.randint(0, len(token_stream)-1)
        if len(token_stream[row_idx]) >= SEQ_LENGTH:
            break


    start_idx = random.randint(0, len(token_stream[row_idx]) - SEQ_LENGTH)

    return token_stream[row_idx][start_idx:start_idx + SEQ_LENGTH]


def verify_sequence(three_grams, seq):
    """Not a true verification; only checks 3-grams."""
    for i in range(len(seq) - 3):
        if tuple(seq[i:i + 3]) not in three_grams:
            return False
    print("***!!! True ", tuple(seq[i:i + 3]))
    return True


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    token_stream, all_tracks = get_data()
    assert START_TOKEN == 0
    track_keys = []
    for key in all_tracks.keys():
        track_keys.append(key)
    words = ['_START'] + list
    print(words)

    print("At type: ",type(all_tracks))
    idx2word = {}
    print("LEN: ", len(all_tracks))
    for i in range(len(all_tracks)):

        idx2word[i] = all_tracks.get(i)['track_name']
    # word2idx = dict((word, i) for i, word in enumerate(words))
    #print("W2I: ", idx2word)
    num_words = len(words)
    three_grams = {}
    count=0
    sec_count=0
    for idx, row in token_stream.iteritems():
        sec_count += len(row)
        if len(row) > 3 :
            for i in range(len(row) - 3):
                three_grams[tuple(w for w in row[i:i + 3])] = True
        else : count += len(row)
    # three_grams = dict((tuple(word2idx[w] for w in token_stream[i:i + 3]), True)
    #                    for i in range(len(token_stream) - 3))
    #print("3Grams ", three_grams)
    print("Less than |3| = ", count)
    print("Total count = ", sec_count)
    print('num words', num_words)
    print('stream length', len(token_stream))
    print('distinct 3-grams', len(three_grams))

    trainable_model = get_trainable_model(num_words)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    print('training')
    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch', epoch)
        proportion_supervised = max(0.0, 1.0 - CURRICULUM_RATE * epoch)
        train.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=1, d_steps=D_STEPS,
            next_sequence=lambda: get_random_sequence(token_stream),
            verify_sequence=lambda seq: verify_sequence(three_grams, seq),
            words=words)

        print("Time taken: ", time.time()-start_time)
        start_time = time.time()

if __name__ == '__main__':
    main()
