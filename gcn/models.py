from gcn.layers import GraphConvolution, FullyConnected
from gcn.metrics import masked_accuracy, masked_softmax_cross_entropy
import tensorflow as tf
from copy import copy
from gcn.utils import recursive_map


class IGCN(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = [tf.nn.relu for i in range(len(self.model_config['connection']))]
        self.act[-1] = lambda x: x

        self._placeholders = placeholders
        self.assign_data = []

        def wrap_placeholder(placeholder):
            if isinstance(placeholder, tf.Tensor):
                dtype = placeholder.dtype
                shape = placeholder.shape
                tensor = tf.Variable(0,
                                     name=placeholder.name.split(':')[0], trainable=False, dtype=placeholder.dtype,
                                     expected_shape=shape, validate_shape=False, collections=[])
                tensor.set_shape(shape)
                self.assign_data.append(tf.assign(tensor, placeholder, validate_shape=False))
                return tensor
            elif isinstance(placeholder, tf.SparseTensor):
                return tf.SparseTensor(indices=wrap_placeholder(placeholder.indices),
                                       values=wrap_placeholder(placeholder.values),
                                       dense_shape=wrap_placeholder(placeholder.dense_shape))

        self.placeholders = recursive_map(placeholders, wrap_placeholder)
        self.assign_data = tf.group(self.assign_data)

        # self.placeholders = placeholders
        self.inputs = self.placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = self.placeholders['labels'].get_shape().as_list()[1]
        self.outputs = None

        self.global_step = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = model_config['optimizer'](learning_rate=self.model_config['learning_rate'])

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution, 'f': FullyConnected}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        sparse = isinstance(self.placeholders['features'], tf.SparseTensor)
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.activations.append(self.inputs)
            for output_dim, layer_cls, act, conv_config in zip(layer_size[1:], layer_type, self.act,
                                                               self.model_config['conv_config']):
                # create Variables
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             conv_config=conv_config))
                sparse = False
                # Build sequential layer model
                hidden = self.layers[-1]()  # build the graph, give layer inputs, return layer outpus
                self.activations.append(hidden)

            self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summary = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)

        self.cross_entropy_loss = masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                               self.placeholders['labels_mask'])
        self.loss += self.cross_entropy_loss

    def _laplacian_regularization(self, graph_signals):
        self.lapla_reg = tf.sparse_tensor_dense_matmul(self.placeholders['laplacian'], graph_signals)
        self.lapla_reg = tf.matmul(tf.transpose(graph_signals), self.lapla_reg)
        self.lapla_reg = tf.trace(self.lapla_reg)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _accuracy_of_class(self):
        self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'] * self.placeholders['labels'][:, i])
                                  for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
