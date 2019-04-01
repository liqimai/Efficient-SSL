from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
from scipy import sparse
from os import path
from gcn.utils import (construct_feed_dict, preprocess_adj, load_data, sparse_to_tuple)
from gcn.lp import Model17
from gcn.graphconv import graphconv
from gcn.models import IGCN
from config import configuration, args


# from gcn import metrics

def train(model_config, sess, repeat_state):
    # Print model_name
    very_begining = time.time()
    print('',
          'name           : {}'.format(model_config['name']),
          'dataset        : {}'.format(model_config['dataset']),
          sep='\n')

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        load_data(model_config['dataset'], train_size=model_config['train_size'],
                  validation_size=model_config['validation_size'],
                  model_config=model_config, shuffle=model_config['shuffle'], repeat_state=repeat_state)

    if model_config['Model'] == 'LP':
        train_time = time.time()
        test_acc, test_acc_of_class = Model17(adj, model_config['alpha'], y_train, y_test)
        train_time = time.time() - train_time
        print("Test set results: accuracy= {:.5f}".format(test_acc))
        print("Total time={}s".format(time.time() - very_begining))
        return test_acc, test_acc_of_class, 0, train_time, train_time

    # preprocess_features
    if model_config['smooth_config']['type'] is not None:
        if model_config['connection'] == ['f' for i in range(len(model_config['connection']))]:
            fetch = train_mask + val_mask + test_mask
            new_features = np.zeros(features.shape, dtype=features.dtype)
            new_features[fetch], smoothing_time = graphconv(features, adj, model_config['smooth_config'], fetch=fetch)
            features = new_features
        else:
            features, smoothing_time = graphconv(features, adj, model_config['smooth_config'])
    else:
        smoothing_time = 0

    support = [preprocess_adj(adj)]
    num_supports = 1

    # Speed up for MLP
    is_mlp = model_config['connection'] == ['f' for _ in range(len(model_config['connection']))]
    if is_mlp:
        train_features = features[train_mask]
        y_train = y_train[train_mask]
        y_train = y_train.astype(np.int32)

        val_features = features[val_mask]
        test_features = features[test_mask]
        labels_mask = np.ones(train_mask.sum(), dtype=np.int32)
    else:
        train_features = features
        val_features = features
        test_features = features
        labels_mask = train_mask.astype(np.int32)
        y_train = y_train.astype(np.int32)

    input_dim = features.shape[1]
    if sparse.issparse(features):
        train_features = sparse_to_tuple(train_features)
        val_features = sparse_to_tuple(val_features)
        test_features = sparse_to_tuple(test_features)
        features = sparse_to_tuple(features)

    # Define placeholders
    placeholders = {
        'labels': tf.placeholder_with_default(y_train, name='labels', shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder_with_default(labels_mask, shape=(None), name='labels_mask'),
        'dropout': tf.placeholder_with_default(0., name='dropout', shape=()),
        'adj_nnz': tf.placeholder_with_default(support[0].values.shape, shape=(1), name='adj_nnz'),
    }
    if not is_mlp:
        placeholders['support'] = [tf.sparse_placeholder(tf.float32, name='support' + str(i)) for i in
                                   range(num_supports)]
    if isinstance(train_features, tf.SparseTensorValue):
        placeholders['num_features_nonzero'] = tf.placeholder_with_default(train_features[1].shape,
                                                                           shape=(1), name='num_features_nonzero')
        placeholders['features'] = tf.sparse_placeholder(tf.float32, name='features')
    else:
        placeholders['num_features_nonzero'] = tf.placeholder_with_default([0],
                                                                           shape=(1), name='num_features_nonzero')
        placeholders['features'] = tf.placeholder_with_default(train_features, shape=[None, features.shape[1]],
                                                               name='features')

    # Create model
    model = IGCN(model_config, placeholders, input_dim=input_dim)

    # Random initialize
    sess.run(tf.global_variables_initializer())

    # Initialize FileWriter, saver & variables in graph
    train_writer = None
    valid_writer = None
    saver = tf.train.Saver()

    # Construct feed dictionary
    if is_mlp:
        if isinstance(features, tf.SparseTensorValue):
            train_feed_dict = {
                placeholders['features']: train_features,
                placeholders['dropout']: model_config['dropout'],
            }
        else:
            train_feed_dict = {placeholders['dropout']: model_config['dropout']}

        valid_feed_dict = construct_feed_dict(
            val_features, support, y_val[val_mask],
            np.ones(val_mask.sum(), dtype=np.bool), 0, placeholders)

        test_feed_dict = construct_feed_dict(
            test_features, support, y_test[test_mask],
            np.ones(test_mask.sum(), dtype=np.bool), 0, placeholders)
    else:
        train_feed_dict = construct_feed_dict(train_features, support, y_train, train_mask, model_config['dropout'],
                                              placeholders)
        valid_feed_dict = construct_feed_dict(val_features, support, y_val, val_mask, 0, placeholders)
        test_feed_dict = construct_feed_dict(test_features, support, y_test, test_mask, 0, placeholders)

    # Some support variables
    acc_list = []
    max_valid_acc = 0
    min_train_loss = 1000000
    t_test = time.time()

    sess.run(model.assign_data, feed_dict=test_feed_dict)
    test_cost, test_acc, test_acc_of_class = sess.run(
        [model.cross_entropy_loss, model.accuracy, model.accuracy_of_class])
    sess.run(model.assign_data, feed_dict=train_feed_dict)

    valid_loss, valid_acc, valid_summary = sess.run([model.cross_entropy_loss, model.accuracy, model.summary],
                                                    feed_dict=valid_feed_dict)
    test_duration = time.time() - t_test
    train_time = 0

    step = model_config['epochs']
    if model_config['train']:
        # Train model
        print('training...')
        for step in range(model_config['epochs']):

            # Training step
            t = time.time()
            sess.run(model.opt_op)
            t = time.time() - t
            train_time += t

            train_loss, train_acc = sess.run([model.cross_entropy_loss, model.accuracy])

            # if True:
            if step > model_config['epochs'] * 0.9 or step % 20 == 0:
                # If it's best performence so far, evalue on test set
                if model_config['validate']:
                    sess.run(model.assign_data, feed_dict=valid_feed_dict)
                    valid_loss, valid_acc = sess.run([model.cross_entropy_loss, model.accuracy])

                    acc_list.append(valid_acc)
                    if valid_acc >= max_valid_acc:
                        max_valid_acc = valid_acc

                        t_test = time.time()
                        sess.run(model.assign_data, feed_dict=test_feed_dict)
                        test_cost, test_acc, test_acc_of_class = \
                            sess.run([model.cross_entropy_loss, model.accuracy, model.accuracy_of_class])
                        test_duration = time.time() - t_test
                        if args.verbose:
                            print('*', end='')
                else:
                    acc_list.append(train_acc)
                    if train_loss < min_train_loss:
                        min_train_loss = train_loss
                        t_test = time.time()
                        sess.run(model.assign_data, feed_dict=test_feed_dict)
                        test_cost, test_acc, test_acc_of_class = \
                            sess.run([model.cross_entropy_loss, model.accuracy, model.accuracy_of_class])
                        test_duration = time.time() - t_test
                        if args.verbose:
                            print('*', end='')
                sess.run(model.assign_data, feed_dict=train_feed_dict)

            # Print results
            if args.verbose:
                print("Epoch: {:04d}".format(step),
                      "train_loss= {:.3f}".format(train_loss),
                      "train_acc= {:.3f}".format(train_acc), end=' ')
                if model_config['validate']:
                    print(
                        "val_loss=", "{:.3f}".format(valid_loss),
                        "val_acc= {:.3f}".format(valid_acc), end=' ')
                else:
                    print(
                        "test_loss=", "{:.3f}".format(test_cost),
                        "test_acc= {:.3f}".format(test_acc), end=' ')
                print("time=", "{:.5f}".format(t))

        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

        # Saving
        if model_config['logdir']:
            print('Save model to "{:s}"'.format(saver.save(
                sess=sess,
                save_path=path.join(model_config['logdir'], 'model.ckpt'))))
    print("Total time={}s".format(time.time() - very_begining))

    return test_acc, test_acc_of_class, train_time / step * 1000, smoothing_time, train_time + smoothing_time


if __name__ == '__main__':

    acc = [[] for i in configuration['model_list']]
    acc_of_class = [[] for i in configuration['model_list']]
    duration = [[] for i in configuration['model_list']]
    smoothing_times = [[] for i in configuration['model_list']]
    total_train_times = [[] for i in configuration['model_list']]
    # Read configuration
    # try:
    for r in range(configuration['repeating']):
        for model_config, i in zip(configuration['model_list'], range(len(configuration['model_list']))):
            # Set random seed
            seed = model_config['random_seed']
            np.random.seed(seed)
            model_config['random_seed'] = np.random.random_integers(1073741824)

            # Initialize session
            with tf.Graph().as_default():
                tf.set_random_seed(seed)
                gpu_options = tf.GPUOptions(allow_growth=True)
                with tf.Session(config=tf.ConfigProto(
                        intra_op_parallelism_threads=model_config['threads'],
                        inter_op_parallelism_threads=2,  # model_config['threads'],
                        gpu_options=gpu_options)) as sess:
                    test_acc, test_acc_of_class, t, smoothing_time, total_train_time = train(
                        model_config, sess, r)
                    acc[i].append(test_acc)
                    acc_of_class[i].append(test_acc_of_class)
                    duration[i].append(t)
                    smoothing_times[i].append(smoothing_time)
                    total_train_times[i].append(total_train_time)
                    if args.verbose:
                        print("accuracy of each class=", test_acc_of_class)
            print('repeated ', r, 'rounds')

    acc_means = np.mean(acc, axis=1)
    acc_stds = np.std(acc, axis=1)
    # acc_of_class_means = np.mean(np.array(acc_of_class), axis=1)
    duration = np.mean(duration, axis=1)
    smoothing_times = np.mean(smoothing_times, axis=1)
    total_train_times = np.mean(total_train_times, axis=1)
    # print mean, standard deviation, and model name
    print()
    for line, model_config in zip(acc, configuration['model_list']):
        print(' '.join('{:.6f}'.format(j) for j in line), model_config['name'])
    # pprint.pprint(acc)
    print("REPEAT\t{}".format(configuration['repeating']))
    print("{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format('DATASET', 'train_size', 'valid_size',
                                                                          'RESULTS', 'STD', 'TIME/STEP(ms)',
                                                                          'SMOOTHING(s)', 'TOTAL_TIME(s)', 'NAME'))
    for model_config, acc_mean, acc_std, t, smoothing_time, total_train_time in zip(configuration['model_list'],
                                                                                    acc_means, acc_stds, duration,
                                                                                    smoothing_times, total_train_times):
        print("{:<8}\t{:<8}\t{:<8}\t{:<8.6f}\t{:<8.6f}\t{:<8.2f}\t{:<8.3f}\t{:<8.3}\t{:<8}".format(
            model_config['dataset'],
            str(model_config['train_size']) + ' / class',
            str(model_config['validation_size']),
            acc_mean,
            acc_std,
            t,
            smoothing_time,
            total_train_time,
            model_config['name']))
