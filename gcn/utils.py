from __future__ import print_function

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import os

from collections import Sequence, Mapping


def recursive_map(root, func):
    if isinstance(root, Sequence):
        return type(root)(recursive_map(item, func) for item in root)
    if isinstance(root, Mapping):
        return type(root)((k, recursive_map(root[k], func)) for k in root)
    else:
        return func(root)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def split_dataset(labels, train_size, test_size, validation_size, validate=True, shuffle=True):
    idx = np.arange(len(labels))
    idx_test = []
    if shuffle:
        np.random.shuffle(idx)
    if isinstance(train_size, int):
        assert train_size > 0, "train size must bigger than 0."
        no_class = labels.shape[1]  # number of class
        train_size = [train_size for i in range(labels.shape[1])]
        idx_train = []
        count = [0 for i in range(no_class)]
        label_each_class = train_size
        next = 0
        for i in idx:
            if count == label_each_class:
                break
            next += 1
            for j in range(no_class):
                if labels[i, j] and count[j] < label_each_class[j]:
                    idx_train.append(i)
                    count[j] += 1
                    break
            else:
                idx_test.append(i)

        idx_test = np.array(idx_test, dtype=idx.dtype)
        if validate:
            if validation_size:
                assert next + validation_size < len(idx), "Too many train data, no data left for validation."
                idx_val = idx[next:next + validation_size]
                next = next + validation_size
            else:
                idx_val = idx[next:]

            assert next < len(idx), "Too many train and validation data, no data left for testing."
            if test_size:
                assert next + test_size < len(idx)
                idx_test = idx[-test_size:]
            else:
                idx_test = np.concatenate([idx_test, idx[next:]])
        else:
            assert next < len(idx), "Too many train data, no data left for testing."
            if test_size:
                assert next + test_size < len(idx)
                idx_test = idx[-test_size:]
            else:
                idx_test = np.concatenate([idx_test, idx[next:]])
            idx_val = idx_test
    else:
        # train
        assert isinstance(train_size, float)
        assert 0 < train_size < 1, "float train size must be between 0-1"
        labels_of_class = [0]
        train_size = int(len(idx) * train_size)
        next = 0
        try_time = 0
        while np.prod(labels_of_class) == 0 and try_time < 100:
            np.random.shuffle(idx)
            idx_train = idx[next:next + train_size]
            labels_of_class = np.sum(labels[idx_train], axis=0)
            try_time = try_time + 1
        next = train_size

        # validate
        if validate:
            assert isinstance(validation_size, float)
            validation_size = int(len(idx) * validation_size)
            idx_val = idx[next: next + validation_size]
            next += validation_size
        else:
            idx_val = idx[next:]

        # test
        if test_size:
            assert isinstance(test_size, float)
            test_size = int(len(idx) * test_size)
            idx_test = idx[next: next + test_size]
        else:
            idx_test = idx[next:]
    idx_train = np.array(idx_train)
    return idx_train, idx_val, idx_test


def load_data(dataset_str, train_size, validation_size, model_config, shuffle=True, repeat_state=None):
    if train_size == 'public':
        return load_public_split_data(dataset_str)
    """Load data."""
    if dataset_str in ['large_cora']:
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0], np.max(l) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']
        adj = data['G']
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        adj = nx.to_scipy_sparse_matrix(nx.from_dict_of_lists(graph))
        # adj = sp.csr_matrix(adj)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))

        if dataset_str.startswith('nell'):
            # Find relation nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(allx.shape[0], len(graph))
            isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
            tx_extended[test_idx_range - allx.shape[0], :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
            ty_extended[test_idx_range - allx.shape[0], :] = ty
            ty = ty_extended

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            labels = np.vstack((ally, ty))
            labels[test_idx_reorder, :] = labels[test_idx_range, :]

            idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

            if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
                print("Creating feature vectors for relations - this might take a while...")
                features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                              dtype=np.int32).todense()
                features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
                features = sp.csr_matrix(features_extended, dtype=np.float32)
                print("Done!")
                save_sparse_csr("data/{}.features".format(dataset_str), features)
            else:
                features = load_sparse_csr("data/{}.features.npz".format(dataset_str))
            idx_train = np.arange(x.shape[0])
            idx_test = test_idx_reorder
            if model_config['validate']:
                assert x.shape[0] + validation_size < allx.shape[0] + tx.shape[0]
                idx_val = np.arange(x.shape[0], x.shape[0] + validation_size)
            else:
                idx_val = test_idx_reorder

            train_mask = sample_mask(idx_train, labels.shape[0])
            val_mask = sample_mask(idx_val, labels.shape[0])
            test_mask = sample_mask(idx_test, labels.shape[0])

            y_train = np.zeros(labels.shape)
            y_val = np.zeros(labels.shape)
            y_test = np.zeros(labels.shape)
            y_train[train_mask, :] = labels[train_mask, :]
            y_val[val_mask, :] = labels[val_mask, :]
            y_test[test_mask, :] = labels[test_mask, :]

            return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        features = preprocess_features(features, feature_type=model_config['feature'])

    # split the data set
    idx_train, idx_val, idx_test = split_dataset(labels, train_size, model_config['test_size'], validation_size,
                                                 validate=model_config['validate'], shuffle=shuffle)

    if model_config['verbose']:
        print('labels of each class : ', np.sum(labels[idx_train], axis=0))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)

    features = features.astype(np.float32)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_public_split_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)
    adj = sp.csr_matrix(adj)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return tf.SparseTensorValue(coords, values, np.array(shape, dtype=np.int64))

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, feature_type):
    if feature_type == 'bow':
        # """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        # normalize(features, norm='l1', axis=1, copy=False)
    elif feature_type == 'tfidf':
        transformer = TfidfTransformer()
        features = transformer.fit_transform(features)
    elif feature_type == 'none':
        features = sp.csr_matrix(sp.eye(features.shape[0]))
    else:
        raise ValueError('Invalid feature type: ' + str(feature_type))
    return features


def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, type=type)
    return sparse_to_tuple(adj)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], laplacian))

    return sparse_to_tuple(t_k)


def absorption_probability(W, alpha, stored_A=None, column=None):
    store_dir = 'cache/'
    stored_A = store_dir + stored_A
    try:
        # raise Exception('DEBUG')
        A = np.load(stored_A + str(alpha) + '.npz')['arr_0']
        print('load A from ' + stored_A + str(alpha) + '.npz')
        if column is not None:
            P = np.zeros(W.shape)
            P[:, column] = A[:, column]
            return P
        else:
            return A
    except:
        # W=sp.csr_matrix([[0,1],[1,0]])
        # alpha = 1
        print('Calculate absorption probability...')
        W = W.copy().astype(np.float32)
        D = W.sum(1).flat
        L = sp.diags(D, dtype=np.float32) - W
        # L = L.dot(L)
        L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
        L = sp.csc_matrix(L)
        # print(np.linalg.det(L))

        if column is not None:
            A = np.zeros(W.shape)
            # start = time.time()
            A[:, column] = slinalg.spsolve(L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])).toarray()
            # print(time.time()-start)
            return A
        else:
            # start = time.time()
            A = slinalg.inv(L).toarray()
            # print(time.time()-start)
            if stored_A:
                np.savez(stored_A + str(alpha) + '.npz', A)
            return A


def construct_knn_graph(features, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    adj = nbrs.kneighbors_graph()
    adj = adj + adj.T
    adj[adj != 0] = 1
    return adj


def construct_feed_dict(features, support, labels, labels_mask, dropout, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    if 'support' in placeholders:
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape if isinstance(features,
                                                                                            tf.SparseTensorValue) else [
        0]})
    feed_dict.update({placeholders['dropout']: dropout})
    return feed_dict


def preprocess_model_config(model_config):
    model_config['connection'] = list(model_config['connection'])
    # judge if parameters are legal
    for c in model_config['connection']:
        if c not in ['c', 'f']:
            raise ValueError(
                'connection string specified by --connection can only contain "c" or "f" but "{}" found'.format(
                    c))
    for i in model_config['layer_size']:
        if not isinstance(i, int):
            raise ValueError('layer_size should be a list of int, but found {}'.format(model_config['layer_size']))
        if i <= 0:
            raise ValueError('layer_size must be greater than 0, but found {}' % i)
    if not len(model_config['connection']) == len(model_config['layer_size']) + 1:
        raise ValueError('length of connection string should be equal to length of layer_size list plus 1')

    # Generate name
    if not model_config['name']:
        model_name = ''
        model_name += str(model_config['Model'])
        from gcn import graphconv
        model_name = graphconv.model_name_modify(model_name, model_config['smooth_config'])
        for char, size, conv_config in \
                zip(model_config['connection'], model_config['layer_size'] + [''], model_config['conv_config']):
            if char == 'c':
                model_name += '_' + conv_config['conv']
                if conv_config['conv'] in ['rnm', 'rw']:
                    model_name += str(conv_config['k'])
                if conv_config['conv'] in ['ap']:
                    model_name += str(conv_config['alpha'])
            else:
                model_name += '_' + char
            if size:
                model_name += '_' + str(size)

        if model_config['validate']:
            model_name += '_validate'

        if model_config['Model'] == 'LP':
            model_name = 'LP_' + str(model_config['alpha'])

        model_config['name'] = model_name


if __name__ == '__main__':
    pass
