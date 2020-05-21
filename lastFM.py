import model
import train
import lastFM

import os.path
import numpy as np
import pandas as pd
import pickle
import random
import tensorflow as tf
import time

# HYPERPARAMETERS
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
    if not os.path.isfile('sessions.pkl') or not os.path.isfile('all_tracks.pkl'):
        token_stream, all_tracks = lastFM.get_sessions(100000)
        token_stream.to_pickle("sessions.pkl")
        output = open('all_tracks.pkl', 'wb')
        pickle.dump(all_tracks, output)
    else:
        print("Sessions exist, using sessions.pkl and all_tracks.pkl")
        token_stream = pd.read_pickle("sessions.pkl")
        all_tracks = pd.read_pickle("all_tracks.pkl")
    return token_stream, all_tracks


class BookGRU(model.GRU):

    def d_optimizer(self, *args, **kwargs):
        return tf.compat.v1.train.AdamOptimizer()  # ignore learning rate

    def g_optimizer(self, *args, **kwargs):
        return tf.compat.v1.train.AdamOptimizer()  # ignore learning rate


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
    return True

def init_dict():
    results = {}
    results['d_losses'] = {}
    results['supervised_g_losses'] = {}
    results['unsupervised_g_losses'] = {}
    results['supervised_generations'] = {}
    results['unsupervised_generations'] = {}
    results['rewards'] = {}
    return results


def run():
    tf.compat.v1.disable_eager_execution()
    random.seed(SEED)
    np.random.seed(SEED)
    
    token_stream, all_tracks = get_data() # Read in data & create sessions
    results = init_dict() # initialise results dictionary
    
    # create words from all track_ids
    track_keys = []
    for key in all_tracks.keys():
        track_keys.append(key)
    words = track_keys

    # create index to word dicttionary for track_ids
    idx2word = {}
    for i in range(len(all_tracks)):
        idx2word[i] = all_tracks.get(i)['track_name']
    
    num_words = len(words)
    three_grams = {}
    count=0
    sec_count=0

    # create dictionary of 3-gram verification values
    for idx, row in token_stream.iteritems():
        sec_count += len(row)
        if len(row) > 3 :
            for i in range(len(row) - 3):
                three_grams[tuple(w for w in row[i:i + 3])] = True
        else : count += len(row)
    
    
    # print("Less than |3| = ", count)
    # print("Total count = ", sec_count)
    # print('num words', num_words)
    # print('stream length', len(token_stream))
    # print('distinct 3-grams', len(three_grams))

    trainable_model = get_trainable_model(num_words)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    start_time = time.time()
    print('Training...')
    for D_STEPS in range(1,5):
        for epoch in range(TRAIN_ITER // EPOCH_ITER):
            print('epoch', epoch)
            proportion_supervised = max(0.0, 1.0 - CURRICULUM_RATE * epoch)
            train.train_epoch(
                sess, trainable_model, EPOCH_ITER,
                proportion_supervised=proportion_supervised,
                g_steps=1, d_steps=D_STEPS,
                next_sequence=lambda: get_random_sequence(token_stream),
                verify_sequence=lambda seq: verify_sequence(three_grams, seq),
                words=words, results=results, epoch=epoch)

            print("Time taken: ", time.time()-start_time)
  
        # Save results for each value of D_STEPS
        with open('results_'+str(D_STEPS)+'.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return token_stream, all_tracks

if __name__ == '__main__':
    token_stream, all_tracks = run()
