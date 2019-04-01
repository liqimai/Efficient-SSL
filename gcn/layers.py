from gcn.inits import *
import tensorflow as tf


def sparse_dropout(x, keep_prob, noise_shape, rescale=True):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    if rescale:
        return pre_out * (1. / keep_prob)
    else:
        return pre_out


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y, a_is_sparse=True)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, input, **kwargs):
        allowed_kwargs = {'name', 'logging', 'use_theta', 'k', 'conv_config'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.input = input
        if isinstance(self.input, tf.SparseTensor):
            self.input_dim = self.input._my_input_dim
        else:
            self.input_dim = self.input.get_shape()[1].value

    def _call(self):
        return self.input

    def __call__(self):
        with tf.name_scope(self.name + '_cal'):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', self.input)
            outputs = self._call()
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        pass
        # for var in self.vars:
        #     tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class FullyConnected(Layer):
    """Fully Connected Layer."""

    def __init__(self, input, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.output_dim = output_dim
        input_dim = self.input_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.name_scope(self.name):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self):
        x = self.input

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)
        self.dropout_x = x

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    def get_lsoftmax_wnorm(self):
        if self.bias:
            w = tf.concat([self.vars['weights'], tf.reshape(self.vars['bias'], [1, -1])], axis=0)
        else:
            w = self.vars['weights']
        return tf.norm(w, axis=0, keep_dims=True)

    def get_lsoftmax_xnorm(self):
        xnorm = tf.reduce_sum(tf.pow(self.dropout_x, 2.0), axis=1, keep_dims=True)
        if self.bias:
            xnorm += 1.0
        xnorm = tf.sqrt(xnorm)  # (batch_size,)
        return xnorm


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 conv_config=1, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        input_dim = self.input_dim
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        # self.featureless = featureless
        self.bias = bias
        self.conv_config = conv_config
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.adj_nnz = placeholders['adj_nnz']

        with tf.name_scope(self.name):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def conv(self, adj, features):
        def tf_ap_approximate(adj, features, alpha):
            alpha = 1.0 / alpha
            k = int(np.ceil(4 / alpha))

            adj = adj / (alpha + 1)
            new_feature = features
            for _ in range(k):
                new_feature = tf.sparse_tensor_dense_matmul(adj, new_feature)
                new_feature += features
            new_feature *= alpha / (alpha + 1)
            return new_feature

        def tf_rnm(adj, features, k):
            new_feature = features
            for _ in range(k):
                new_feature = tf.sparse_tensor_dense_matmul(adj, new_feature)
            return new_feature

        def tf_rw(adj, features, k):
            new_feature = features
            for _ in range(k):
                new_feature = (tf.sparse_tensor_dense_matmul(adj, new_feature) + new_feature) / 2
            return new_feature

        c = self.conv_config['conv']
        assert c in ['rnm', 'rw', 'ap']
        if c == 'rnm':
            result = tf_rnm(adj, features, self.conv_config['k'])
        elif c == 'rw':
            result = tf_rw(adj, features, self.conv_config['k'])
        elif c == 'ap':
            result = tf_ap_approximate(adj, features, self.conv_config['alpha'])
        return result

    def _call(self):
        x = self.input

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)
        self.dropout_x = x

        # convolve
        supports = list()
        H = None
        for i in range(len(self.support)):
            pre_sup = dot(x, self.vars['weights_' + str(i)],
                          sparse=self.sparse_inputs)
            support = pre_sup
            support = self.conv(
                self.support[i],  # sparse_dropout(self.support[i], 1 - self.dropout, self.adj_nnz, rescale=False),
                support)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
