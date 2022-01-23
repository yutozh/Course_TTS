import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

class CBHG:
    def __init__(self, K=16, depth=256):
        self.K = K
        self.depth = depth

    def __call__(self, inputs, input_lengths, is_training, scope):
        K = self.K
        depth = self.depth
        input_channel = inputs.get_shape()[2]
        projections = [128, input_channel]

        with tf.variable_scope(scope):
            with tf.variable_scope('conv_bank'):
                # Convolution bank: concatenate on the last axis to stack channels from all convolutions
                conv_outputs = tf.concat(
                    [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
                    axis=-1
                )

            # Maxpooling:
            maxpool_output = tf.layers.max_pooling1d(
                conv_outputs,
                pool_size=2,
                strides=1,
                padding='same')

            # Two projection layers:
            proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
            proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

            # Residual connection:
            highway_input = proj2_output + inputs

            half_depth = depth // 2
            assert half_depth*2 == depth, 'encoder and postnet depths must be even.'

            # Handle dimensionality mismatch:
            if highway_input.shape[2] != half_depth:
                highway_input = tf.layers.dense(highway_input, half_depth)

            # 4-layer HighwayNet:
            for i in range(4):
                highway_input = highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
            rnn_input = highway_input

            # Bidirectional RNN
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                GRUCell(half_depth),
                GRUCell(half_depth),
                rnn_input,
                sequence_length=input_lengths,
                dtype=tf.float32)
            return tf.concat(outputs, axis=2)  # Concat forward and backward

def highwaynet(inputs, scope, depth):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)

class CBHG2:
    def __init__(self, batch_size, K=16, embed_size=128, ):
        self.K = K
        self.embed_d = embed_size
        self.batch_size = batch_size
        self.lstm_hidden_size = 64

    def __call__(self, inputs, input_length):
        '''

        :param inputs:  (batch_size, sequence_size, embedding_size)
        :param input_length:
        :return:
        '''
        input_x = tf.expand_dims(inputs, -1)  ## b,T,d,1
        self.input = input_x

        input_x = tf.transpose(self.input, [0, 2, 1, 3])  ## b,d,T,1

        conv_list = []
        previous_shape = [d.value for d in input_x.get_shape().dims]
        for k in range(1, self.K + 1):
            conv = self.convolution(input_x, kh=previous_shape[1], kw=k, filter_in=1, filter_out=64, layer=1)
            conv_list.append(tf.squeeze(conv, axis=1))
        conv0 = tf.concat(conv_list, axis=2)  # b,T,filter_out*self.K
        conv0 = tf.expand_dims(tf.transpose(conv0, [0, 2, 1]), -1)  # b,filter_out*self.K , T, 1

        k_size = [1, 1, 2, 1]
        max_pool = self.pooling(conv0, k_size, strides=[1, 1, 1, 1], layer=2)

        pool_shape = [d.value for d in max_pool.get_shape().dims]
        conv1 = self.convolution(max_pool, kh=self.K * 64, kw=3,
                                 filter_in=1, filter_out=self.embed_d, layer=3)
        conv1 = tf.squeeze(conv1, axis=1)  # b,T,d

        conv1d_project = tf.expand_dims(conv1, -1)
        resnet = self.input + conv1d_project  # b,T,d = embed_d,1
        highway_net = self.highway(input_=resnet, size=[self.embed_d, self.embed_d], num_layers=3, f=tf.nn.relu, scope='Highway')

        # b, T, (d = embed_d) , 1
        ## notice since num_layers of network consecutive, size = [in,out] must be same above
        bi_rnn = self.Bi_RNN(tf.squeeze(highway_net, -1))  # [batch_size, max_time, hidden_size]

        return bi_rnn

    def convolution(self, input_x, kh, kw, filter_in, filter_out, layer):
        '''

        :param input_x:
        :param kh:
        :param kw:
        :param filter_in:
        :param filter_out:
        :param layer:
        :return:  b,1,T,filter_out(64)
        '''
        ## 1D-conv, so kw = D input_dim
        ## kw = k, where k = 1 ... 16
        filter_shape = [kh, kw, filter_in, filter_out]

        with tf.variable_scope("conv-%i-%i" % (layer, kw)) as scope:
            W = tf.get_variable(name='W-%i' % layer,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32),
                                shape=filter_shape)
            b = tf.get_variable(name='b-%i' % layer,
                                initializer=tf.constant_initializer(0.1),
                                shape=[filter_out], dtype=tf.float32)

            ### why not "SAME" work here
            left = (kw - 1) // 2
            right = kw - 1 - left
            input_x = tf.pad(input_x, [[0, 0], [0, 0], [left, right], [0, 0]])
            conv = tf.nn.conv2d(input=input_x,
                                filter=W,
                                strides=[1, 1, 1, 1],  ## kernel size changes with k, but not stride!!!
                                padding='VALID',
                                name="conv-%i-%i" % (layer, kw))
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
        return h

    def pooling(self, input_x, k_size, strides, layer):
        with tf.name_scope('Pooling_{0}'.format(layer)) as scope:
            pool = tf.nn.max_pool(value=input_x, ksize=k_size,
                                  strides=strides,
                                  padding='SAME')
        return pool

    def highway(self, input_, size, num_layers=1, f=tf.nn.relu, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

        size = [in_dim,out_dim]
        """
        layers = 0
        n_in, n_out = size
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                layers += 1
                g = f(self.dense(input_, in_dim=n_in, out_dim=n_out,
                                 layer=layers))
                t = tf.sigmoid(self.dense(input_, in_dim=size[0], out_dim=size[1],
                                          layer=layers))
                output = t * g + (1. - t) * input_
                input_ = output

        return output

    def dense(self, input_x, in_dim, out_dim, layer, scopes=None):
        '''
        input:
            input_x: (batch, T, D, 1)
            input_size: D*T*1
        output:
            (batch, output_unit, T, 1)
        '''

        x_flat = tf.squeeze(input_x, -1)
        ### if change to variable scope: Then, outside this function,
        ### if do not specify this variable scope, we cannot access it
        with tf.variable_scope(scopes or 'dense-%i' % layer) as scope:
            try:
                matrix = tf.get_variable("Matrix", [in_dim, out_dim], dtype=tf.float32)
                bias_term = tf.get_variable('bias_term', [out_dim], dtype=tf.float32)
            except ValueError:
                scope.reuse_variables()
                matrix = tf.get_variable('Matrix')
                bias_term = tf.get_variable('bias_term')
        matrix_b = tf.tile(tf.expand_dims(matrix, 0), [self.batch_size, 1, 1])
        bias = tf.tile(tf.expand_dims(bias_term, 0), [self.batch_size, 1])
        bias = tf.expand_dims(bias, -1)
        dense = tf.matmul(x_flat, matrix_b)
        return tf.expand_dims(dense, -1)

    def Bi_RNN(self, input_x):
        fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)
        # bw_cell = tf.contrib.rnn.LSTMCell(128, forget_bias=1.0, state_is_tuple=True)

        out, states = tf.nn.dynamic_rnn(fw_cell, input_x, dtype=tf.float32)  ## this step does create variables
        #         h, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_cell,
        #                                                      cell_bw = bw_cell,
        #                                                      inputs = input_x,
        #                                                      sequence_length=self.seqlen,
        #                                                      dtype=tf.float32)
        #         out = tf.concat(h,axis = 2)
        return out
