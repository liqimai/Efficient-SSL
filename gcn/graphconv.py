from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import warnings
import time
from gcn.utils import normalize_adj

cupy_is_available = False
try:
    import cupy as cp

    cupy_is_available = True
    pass
except ImportError:
    warnings.warn("cupy is not imported, some operations may not use GPU.", ImportWarning)


def graphconv(features, adj, filter_config, fetch=None):
    smoothing = filter_config['type']
    print(smoothing, 'Smoothing...')
    if smoothing is None:
        return features, 0.
    elif smoothing == 'rnm':
        return taubin_smoothing(adj, 1, 0, filter_config['k'], features, fetch=fetch)
    elif smoothing == 'rw':
        return taubin_smoothing(adj, 0.5, 0, filter_config['k'], features, fetch=fetch)
    elif smoothing == 'taubin':
        return taubin_smoothing(adj, filter_config['taubin_lambda'], filter_config['taubin_mu'],
                                filter_config['taubin_repeat'], features, fetch=fetch)
    elif smoothing == 'ap_appro':
        alpha = filter_config['alpha']
        k = int(np.ceil(4 * alpha))
        return ap_approximate(adj, features, filter_config['alpha'], k, fetch=fetch)
    else:
        raise ValueError("graphconv must be one of 'ap' and 'taubin' ")


def gpu_taubin_smoothing(step_transformor, features, repeat, fetch):
    # TODO: transfer sparse features to GPU
    # TODO: only fetch necessary data
    smooth_time = 0
    step_transformor = cp.sparse.csr_matrix(step_transformor)
    step_transformor.sum_duplicates()
    tile_width = 1024 ** 3 // 4 // 4 // features.shape[0]
    # initialzie new_features
    if fetch is None:
        new_features = features
    else:
        new_features = features[fetch]
    if sp.issparse(new_features):
        new_features = new_features.todense()

    for i in range(0, features.shape[1], tile_width):
        low = i
        high = min(features.shape[1], i + tile_width)
        # transfer data to GPU
        if sp.issparse(features):
            tile = cp.sparse.csr_matrix(features[:, low:high])
            tile = tile.todense()
        else:
            tile = cp.asarray(features[:, low:high])
        tile = cp.asfortranarray(tile)
        tile.device.synchronize()

        # calculate
        begin = time.time()
        for i in range(repeat):
            tile = cp.cusparse.csrmm2(step_transformor, tile, tile)
            # tile = step_transformor.dot(tile)
        tile.device.synchronize()
        smooth_time += time.time() - begin

        # fetch
        if fetch is None:
            new_features[:, low:high] = tile.get()
        else:
            new_features[:, low:high] = tile[fetch].get()
    return new_features, smooth_time


def taubin_smoothing(adj, lam, mu, repeat, features, fetch=None):
    smooth_time = 0
    n = adj.shape[0]
    # adj = normalize(adj + sp.eye(adj.shape[0]), norm='l1', axis=1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]), 'sym')
    smoothor = sp.eye(n) * (1 - lam) + lam * adj
    inflator = sp.eye(n) * (1 - mu) + mu * adj
    step_transformor = smoothor * inflator
    step_transformor = step_transformor.astype(np.float32)
    features = features.astype(np.float32)
    if cupy_is_available:
        print('USE GPU')
        features, smooth_time = gpu_taubin_smoothing(step_transformor, features, repeat, fetch)
    else:
        if sp.issparse(features):
            features = features.toarray()
        begin = time.time()
        for i in range(repeat):
            features = step_transformor.dot(features)
        smooth_time += time.time() - begin
        if fetch is not None:
            features = features[fetch]
    if sp.issparse(features):
        features = features.toarray()
    return features, smooth_time


def taubin_smoothor(adj, lam, mu, repeat):
    n = adj.shape[0]
    # adj = normalize(adj + sp.eye(adj.shape[0]), norm='l1', axis=1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]), 'sym')
    smoothor = sp.eye(n) * (1 - lam) + lam * adj
    inflator = sp.eye(n) * (1 - mu) + mu * adj
    step_transformor = smoothor * inflator
    transformor = sp.eye(n)
    base = step_transformor
    while repeat != 0:
        if repeat % 2:
            transformor *= base
        base *= base
        repeat //= 2
        # print(repeat)
    return transformor


def gpu_ap_approximate(adj, features, alpha, k, fetch):
    features = features.astype(np.float32)
    if fetch is None:
        new_features = features
    else:
        new_features = features[fetch]
    if sp.issparse(new_features):
        new_features = new_features.todense()

    smooth_time = 0
    adj = cp.sparse.csr_matrix(adj)
    adj.sum_duplicates()
    tile_width = 1024 ** 3 // 4 // 2 // features.shape[0]
    for i in range(0, features.shape[1], tile_width):
        low = i
        high = min(features.shape[1], i + tile_width)
        # transfer data to GPU
        if sp.issparse(features):
            new_features_tile = cp.sparse.csr_matrix(features[:, low:high])
            features_tile = cp.sparse.csr_matrix(features[:, low:high])
            new_features_tile = new_features_tile.todense()
            features_tile = features_tile.todense()
        else:
            new_features_tile = cp.asarray(features[:, low:high])
            features_tile = cp.asarray(features[:, low:high])
        new_features_tile = cp.asfortranarray(new_features_tile)
        new_features_tile.device.synchronize()

        # calculate
        begin = time.time()
        for _ in range(k - 1):
            # new_feature = adj.dot(new_feature) + features
            new_features_tile = cp.cusparse.csrmm2(adj, new_features_tile, new_features_tile)
            new_features_tile += features_tile
        new_features_tile *= alpha / (alpha + 1)
        new_features_tile.device.synchronize()
        smooth_time += time.time() - begin

        # fetch
        if fetch is None:
            new_features[:, low:high] = new_features_tile.get()
        else:
            new_features[:, low:high] = new_features_tile[fetch].get()
    return new_features, smooth_time


def ap_approximate(adj, features, alpha, k, fetch=None):
    smooth_time = 0
    alpha = 1.0 / alpha
    # adj = normalize(adj + sp.eye(adj.shape[0]), 'l1', axis=1) / (alpha + 1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]), 'sym') / (alpha + 1)
    adj = adj.astype(np.float32)
    if cupy_is_available:
        print('USE GPU')
        new_feature, smooth_time = gpu_ap_approximate(adj, features, alpha, k, fetch=fetch)
    else:
        if sp.issparse(features):
            features = features.toarray()
        features = features.astype(np.float32)
        new_feature = np.zeros(features.shape, dtype=features.dtype)
        begin = time.time()
        for _ in range(k):
            new_feature = adj.dot(new_feature)
            new_feature += features
        new_feature *= alpha / (alpha + 1)
        smooth_time += time.time() - begin
        if fetch is not None:
            new_feature = new_feature[fetch]
    return new_feature, smooth_time


def model_name_modify(model_name, smooth_config):
    smoothing = smooth_config['type']
    if smoothing == 'ap_appro':
        model_name += '_ap_appro_' + str(smooth_config['alpha'])
    elif smoothing == 'taubin':
        model_name += '_taubin' + str(smooth_config['taubin_lambda']) \
                      + '_' + str(smooth_config['taubin_mu']) \
                      + '_' + str(smooth_config['taubin_repeat'])
    elif smoothing == 'rnm':
        model_name += '_rnm_' + str(smooth_config['k'])
    elif smoothing == 'rw':
        model_name += '_rnm_' + str(smooth_config['k'])
    elif smoothing is None:
        pass
    else:
        raise ValueError('invalid smoothing')

    return model_name
