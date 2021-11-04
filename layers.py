from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
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

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        # self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        # if self.sparse_inputs:
        #     x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        # else:
        x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights'])
        else:
            pre_sup = self.vars['weights']
        support = dot(self.support, pre_sup, sparse=False)
        supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        self.outs = self.act(output)
        return self.act(output)

class GraphAttention(Layer):
    def __init__(self,):
        pass

    def call(self,inputs):
        pass

    def attn_head(self, seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, seq_fts)
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.keras.layers.Conv1D(seq, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation


class PathEmbedding(Layer):
    def __init__(self, num_quests, num_paths, num_edges, link_state_dim, path_state_dim, placeholders, flow_hidden=6,
                 act=None,
                 **kwargs):
        super(PathEmbedding, self).__init__(**kwargs)
        self.num_quests = num_quests
        self.num_paths = num_paths
        self.num_edges = num_edges
        self.paths = placeholders['paths']
        self.idx = placeholders['index']
        self.seqs = placeholders['sequences']
        self.link_state_dim = link_state_dim  # + num_quests
        self.path_state_dim = path_state_dim
        # gru cell
        self.path_update = tf.keras.layers.GRUCell(path_state_dim)
        self.path_update.build(tf.TensorShape([None, self.link_state_dim]))

        # dense
        self.act = act
        self.w_f0 = glorot([3, flow_hidden])
        self.w_f1 = glorot([flow_hidden, 1])
        self.b_f0 = tf.Variable(tf.random_normal([flow_hidden], stddev=0.1))
        self.b_f1 = tf.Variable(tf.random_normal([1], stddev=0.1))
        # attention
        self.wq = glorot([path_state_dim, path_state_dim])
        self.wk = glorot([path_state_dim, path_state_dim])
        self.wv = glorot([path_state_dim, path_state_dim])

    def _call(self, inputs):
        # RNN
        h_tild = tf.gather(inputs, self.paths)
        ids = tf.stack([self.idx, self.seqs], axis=1)
        max_len = tf.reduce_max(self.seqs) + 1
        shape = tf.stack([self.num_quests * self.num_paths, max_len, self.link_state_dim])
        lens = tf.math.segment_sum(data=tf.ones_like(self.idx),
                                   segment_ids=self.idx)
        link_inputs = tf.scatter_nd(ids, h_tild, shape)

        hidden_states, last_state = tf.nn.dynamic_rnn(self.path_update,
                                                      link_inputs,
                                                      sequence_length=lens,
                                                      dtype=tf.float32)
        ###
        # # 这里是不加attention的部分
        # last_state = tf.reshape(last_state, [self.num_quests, self.num_paths * self.path_state_dim])
        # last_state = tf.keras.layers.Softmax()(last_state)
        # return last_state
        ###

        key = tf.matmul(hidden_states, self.wk)
        query = tf.matmul(last_state, self.wq)
        value = tf.matmul(hidden_states, self.wv)
        self.att = tf.matmul(key, tf.expand_dims(query, -1))
        self.att = tf.transpose(self.att, [0, 2, 1])
        context = tf.matmul(self.att, value)
        # context = tf.squeeze(context)
        # self.att = tf.matmul(tf.transpose(hidden_states,[0,2,1]),state)
        # context = tf.matmul(tf.transpose(self.att, [0, 2, 1]), state)
        # another model:
        # conv0=tf.keras.layers.Conv1D(2*self.link_state_dim,5,activation='relu')(link_inputs)
        # bn0 = tf.keras.layers.BatchNormalization()(conv0)
        # conv1 = tf.keras.layers.Conv1D(self.link_state_dim,3,activation='relu')(bn0)
        # bn1 = tf.keras.layers.BatchNormalization()(conv1)
        # conv2 = tf.keras.layers.Conv1D(self.path_state_dim, 3,activation='relu')(bn1)
        # # context = tf.keras.layers.Dense(self.path_state_dim)(conv2)
        # context = tf.keras.layers.LSTM(self.path_state_dim)(link_inputs,mask=lens)

        # reshape into path embedding
        path_state = tf.reshape(context, [self.num_quests, self.num_paths * self.path_state_dim])
        path_state = tf.keras.layers.Softmax()(path_state)
        return path_state

class BatchTransform(Layer):
    def __init__(self,batch_size):
        super(BatchTransform,self).__init__()
        self.batch_size = batch_size

    def _call(self, inputs):
        _, seq_len, depth = tf.shape(inputs)
        x = tf.reshape(inputs,(self.batch_size,-1,seq_len,depth))
        return x

class Attention(Layer):
    # self attention
    def __init__(self, num_paths, num_quests):
        super(Attention, self).__init__()
        self.num_quests = num_quests
        self.num_paths = num_paths

        # Trainable parameters
        with tf.name_scope('Attention'):
            self.wq = glorot([num_paths, num_paths], name='attention_q')
            self.wk = glorot([num_paths, num_paths], name='attention_k')
            self.wv = glorot([num_paths, num_paths], name='attention_v')

    def _call(self, inputs):
        key = tf.matmul(inputs, self.wk)
        query = tf.matmul(inputs, self.wq)
        value = tf.matmul(inputs, self.wv)  # [T,D]

        # self.att = tf.matmul(key, tf.transpose(query, perm=[1, 0]))/tf.sqrt(tf.cast(self.num_paths,tf.float32))  # [T,T]
        self.att = tf.matmul(tf.transpose(key, perm=[1, 0]), query) / tf.sqrt(tf.cast(self.num_paths, tf.float32))
        self.att = tf.nn.softmax(self.att, axis=-1)
        # matrix version
        # self.att= tf.nn.softmax(tf.transpose(self.att, perm=[1, 0]),axis=-1)
        # context = tf.matmul(tf.transpose(self.att, perm=[1, 0]), value)
        # context = tf.keras.layers.BatchNormalization()(context)

        context = tf.matmul(value, self.att)
        return context

class Residual(Layer):
    def __init__(self,d_model,num_heads,num_mh,**kwargs):
        self.d_model = d_model
        self.num_heads = num_heads
        super(Residual,self).__init__(**kwargs)
        self.layers = []
        for layer in range(num_mh):
            self.layers.append(MultiHeadAttention(d_model=d_model,num_heads=num_heads))


    def _call(self,inputs):
        shape = tf.shape(inputs)
        inputs = tf.expand_dims(inputs,0)
        x= inputs
        for layer in self.layers:
            x = layer(x)
        output = inputs+x
        output = tf.reshape(shape)
        # output = tf.squeeze(output)
        return output


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        with tf.name_scope('Attention'):
            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)

            self.dense = tf.keras.layers.Dense(d_model)

        # self.attention_weights = None

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """计算注意力权重。
        q, k, v 必须具有匹配的前置维度。
        k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
        虽然 mask 根据其类型（填充或前瞻）有不同的形状，
        但是 mask 必须能进行广播转换以便求和。

        参数:
          q: 请求的形状 == (..., seq_len_q, depth)
          k: 主键的形状 == (..., seq_len_k, depth)
          v: 数值的形状 == (..., seq_len_v, depth_v)
          mask: Float 张量，其形状能转换成
                (..., seq_len_q, seq_len_k)。默认为None。

        返回值:
          输出，注意力权重
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # 缩放 matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 将 mask 加入到缩放的张量上。
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
        # 相加等于1。
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def _call(self, inputs, mask=None):
        # inputs = tf.expand_dims(inputs, 0)
        # to reuse the code
        batch_size = 1

        q = self.wq(inputs)  # (batch_size, seq_len, d_model)
        k = self.wk(inputs)  # (batch_size, seq_len, d_model)
        v = self.wv(inputs)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, self.attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        outputs = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        # outputs = tf.squeeze(outputs)

        return outputs


class Readout(Layer):
    def __init__(self, input_dim, output_dim, act=tf.nn.relu, **kwargs):
        super(Readout, self).__init__(**kwargs)
        self.act = act
        self.w0 = glorot([input_dim, output_dim])
        self.b0 = tf.zeros(output_dim)

    def _call(self, inputs):
        # inputs = tf.squeeze(inputs)
        o = tf.matmul(inputs, self.w0) + self.b0
        return self.act(o)
