from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

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


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class PEM(Model):
    def __init__(self, num_paths, num_quests, num_edges, placeholders, learning_rate=0.005,
                 gcn_input_dim=2, gcn_hidden_dim=16, gcn_output_dim=8,
                 pe_output_dim=4, att_layers_num=4,
                 **kwargs):
        super(PEM, self).__init__(**kwargs)

        self.inputs = placeholders['features']

        # hyper-parameters
        self.gcn_input_dim = gcn_input_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gcn_output_dim = gcn_output_dim
        self.pe_output_dim = pe_output_dim
        self.att_layers_num = att_layers_num

        self.num_paths = num_paths
        self.num_quests = num_quests
        self.num_edges = num_edges

        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.build()

    def _build(self):

        # self.layers.append(tf.keras.layers.Dense(self.gcn_input_dim//2))
        self.layers.append(GraphConvolution(input_dim=self.gcn_input_dim,
                                            output_dim=self.gcn_hidden_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.gcn_hidden_dim,
                                            output_dim=self.gcn_output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

        # self.layers.append(GraphConvolution(input_dim=self.gcn_output_dim,
        #                                     output_dim=self.gcn_output_dim//2,
        #                                     placeholders=self.placeholders,
        #                                     act=lambda x: x,
        #                                     dropout=True,
        #                                     logging=self.logging))

        self.layers.append(PathEmbedding(num_paths=self.num_paths,
                                         num_quests=self.num_quests,
                                         num_edges=self.num_edges,
                                         link_state_dim=self.gcn_output_dim,
                                         path_state_dim=self.pe_output_dim,
                                         placeholders=self.placeholders,
                                         act=tf.nn.relu))

        # self.layers.append(Attention(num_paths=self.pe_output_dim*self.num_paths,
        #                              num_quests=self.num_quests))
        # self.layers.append(Residual(d_model=self.pe_output_dim * self.num_paths,
        #                             num_heads=self.num_paths,
        #                             num_mh=self.att_layers_num))
        for l in range(self.att_layers_num):
            # self.layers.append(MultiHeadAttention(d_model=self.pe_output_dim * self.num_paths,
            #                                       num_heads=self.num_paths))
            self.layers.append(tf.keras.layers.Dense(self.pe_output_dim * self.num_paths))

        self.layers.append(tf.keras.layers.Dense(self.num_paths))
        # self.layers.append(Readout(input_dim=self.pe_output_dim*self.num_paths,
        #                            output_dim=self.num_paths))

    def _loss(self):
        l = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.squeeze(self.outputs), labels=self.placeholders['labels'],axis=1)
        loss = tf.reduce_mean(l)  # modify here
        # l2 normalize
        # loss+= tf.nn.l2_loss(self.vars)
        tf.summary.scalar("loss", loss)
        # self.loss += loss
        self.loss = loss

    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.placeholders['labels'], 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(accuracy_all)
        tf.summary.scalar('accuracy', accuracy)
        self.accuracy = accuracy

    def predict(self):
        return tf.nn.softmax(self.outputs)

    def cal_gradient(self):
        return tf.gradients(self.outputs, self.placeholders['features'])

    def save(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, path + "/model/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = path + "/model/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
