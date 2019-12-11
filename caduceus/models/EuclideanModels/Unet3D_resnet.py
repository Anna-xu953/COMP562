import numpy as np
import tensorflow as tf
import time
import scipy.sparse
import sklearn
import math

from scipy.ndimage import zoom

class Unet3D:
    def __init__(self, params, sess, name):
        self.name = name
        self.sess = sess
        self._params=params
        self.features_root = 32
        self.conv_size = 3

        self.regularization = self._params["training_parameters"]["L2Lambda"]["value"]

        self.regularizers = []

        # Build the computational graph.
        self.build_graph()

    def build_graph(self):
        """Build the computational graph of the model."""
        self.graph = self.sess.graph
        with self.graph.as_default():
            # Inputs.
            M_0 = self._params["training_parameters"]["input_width"]["value"]* self._params["training_parameters"]["input_height"]["value"] * self._params["training_parameters"]["input_channels"]["value"]
            M_1 = self._params["training_parameters"]["output_width"]["value"]* self._params["training_parameters"]["output_height"]["value"] * self._params["training_parameters"]["output_channels"]["value"]
            with tf.name_scope(self.name + '_inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self._params["training_parameters"]["batch_size"]["value"], M_0), 'data')
                self.ph_target = tf.placeholder(tf.float32, (self._params["training_parameters"]["batch_size"]["value"], M_1), 'target')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.ph_phase_train = tf.placeholder(tf.bool, name='phase_train')

            # Model.
            
            #data = tf.reshape(self.ph_data, [self._params["training_parameters"]["batch_size"]["value"], M_0])
            
            self.op_results = self.inference(self.ph_data, self.ph_dropout, self.ph_phase_train)
            #self.op_results = tf.reshape(self.op_results, [self._params["training_parameters"]["batch_size"]["value"], self._params["training_parameters"]["output_width"]["value"], self._params["training_parameters"]["output_height"]["value"], self._params["training_parameters"]["output_channels"]["value"], self.nob])
            self.op_loss, self.op_loss_average = self.loss(self.op_results, self.ph_target)
            self.op_train = self.training(self.op_loss, self._params["training_parameters"]["gene_learn_rate"]["value"],
                    self._params["training_parameters"]["decay_steps"]["value"], self._params["training_parameters"]["decay_rate"]["value"], self._params["training_parameters"]["momentum"]["value"])

        self.op_summary = tf.summary.merge_all()
        self.op_saver = tf.train.Saver(max_to_keep=10)

    def loss(self, results, targets):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope(self.name + 'loss'):
            with tf.name_scope('l1loss'):
                # l1loss = 0
                # for d in range(self.nob):
                #     l1loss = l1loss + tf.losses.absolute_difference(labels = results[:, :, :, :, d], predictions=targets[:, :, :, :, d])

                # l1loss /= float(self.nob)
                l2loss=tf.reduce_mean(tf.abs(results - targets))
                tf.losses.add_loss(l2loss)

            with tf.name_scope('regularization'):
                if len(self.regularizers) > 0:
                    regLoss = 0
                    for r in range(len(self.regularizers)):
                        regLoss += self.regularization * self.regularizers[r]

            if len(self.regularizers) > 0:
                loss = l2loss * self._params["training_parameters"]["geneLambda"]["value"] + regLoss
            else:
                loss = l2loss * self._params["training_parameters"]["geneLambda"]["value"] 

            tf.losses.add_loss(loss)
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/l2loss', l2loss)
            if len(self.regularizers) > 0:
                tf.summary.scalar('loss/regLoss', regLoss)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                if len(self.regularizers) > 0:
                    op_averages = averages.apply([l2loss, regLoss, loss])
                    tf.summary.scalar('loss/avg/l2loss', averages.average(l2loss))
                    tf.summary.scalar('loss/avg/regLoss', averages.average(regLoss))
                    tf.summary.scalar('loss/avg/total', averages.average(loss))
                else:
                    op_averages = averages.apply([l2loss, loss])
                    tf.summary.scalar('loss/avg/l2loss', averages.average(l2loss))
                    tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope(self.name + 'training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=self._params["training_parameters"]["beta1"]["value"], beta2=self._params["training_parameters"]["beta2"]["value"])
            #optimizer = tf.train.RMSPropOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                op_train = optimizer.minimize(loss)
            grads = optimizer.compute_gradients(loss, tf.trainable_variables())
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    tf.summary.histogram(var.op.name + '/var', var)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            # count number of trainable parameters
            no_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print ('Model size: %dK' % (no_params/1000,))
            return op_train

    def inference(self, data, dropout, phase_train, reuse=False):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        results = self._inference(data, dropout, phase_train, reuse=reuse)
        return results

    def _inference(self, x, dropout, phase_train, reuse=False):
        x = tf.reshape(x, [self._params["training_parameters"]["batch_size"]["value"], self._params["training_parameters"]["input_width"]["value"], 
                        self._params["training_parameters"]["input_height"]["value"] , self._params["training_parameters"]["input_channels"]["value"] ])
        x = tf.expand_dims(x, axis = 4)
        x0  = x
        # Encoding path
        layers = 3
        conv_size = self.conv_size
        deconv_size = 2
        pool_stride_size = 2
        pool_kernel_size = 2 # Use a larger kernel

        connection_outputs = []
        for layer in range(layers):
            features = 64#2**layer * self.features_root
            if layer == 0:
                prev = x
            else:
                prev = pool
                
            if layer == 0 :
                conv = self.conv_relu(prev, features, conv_size, 1)
                prev = conv
            # else:
            conv = self.res_conv3d(prev, features, conv_size, is_training=phase_train, scope='encoding' + str(layer))
            connection_outputs.append(conv)
            pool = tf.nn.max_pool3d(conv, ksize=[1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1],
                                    strides=[1, pool_stride_size, pool_stride_size, pool_stride_size, 1],
                                    padding='SAME')
        
        bottom = self.res_conv3d(pool, features, conv_size, is_training=phase_train, scope='bottom')
        #bottom = tf.nn.dropout(bottom, dropout)
        
        # Decoding path
        for layer in range(layers):
            conterpart_layer = layers - 1 - layer
            features = 64#2**conterpart_layer * self.features_root
            if layer == 0:
                prev = bottom
            else:
                prev = conv_decoding
            
            shape = prev.get_shape().as_list()
            deconv_output_shape = [tf.shape(prev)[0], shape[1] * deconv_size, shape[2] * deconv_size,
                                   shape[3] * deconv_size, features]
            deconv = self.deconv3d(prev, deconv_output_shape, deconv_size, is_training=phase_train,
                              scope='decoding' + str(conterpart_layer))
            # cc = self.crop_and_concat(connection_outputs[conterpart_layer], deconv)
            cc = tf.concat([connection_outputs[conterpart_layer], deconv], axis = 4)
            cc = self.conv_relu(cc, features, conv_size, 1, scope='decoding_conv' + str(conterpart_layer))
            conv_decoding = self.res_conv3d(cc, features, conv_size, is_training=phase_train,
                                scope='decoding' + str(conterpart_layer))
            
        with tf.variable_scope('last_layer') as scope:
            # #x = self.conv3d(conv_decoding, 1, 1, is_training=phase_train,
            #                        scope='decoding_last')
            w1 = tf.get_variable('w1', [1, 1, 1, conv_decoding.get_shape()[-1], 1],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1 = tf.nn.conv3d(conv_decoding, w1, strides=[1, 1, 1, 1, 1], padding='SAME')
            b1 = tf.get_variable('b1', [1], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(conv1, b1)
            x = x + x0
        x = tf.squeeze(x, axis = 4)
        
        return tf.reshape(x, [self._params["training_parameters"]["batch_size"]["value"],-1])

    def conv3d(self, input_, output_dim, f_size, is_training, scope='conv3d'):
        with tf.variable_scope(scope) as scope:
            # VGG network uses two 3*3 conv layers to effectively increase receptive field
            w1 = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1)) 
                                
            conv1 = tf.nn.conv3d(input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME') # NDHWC format
            b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.bias_add(conv1, b1)
            # bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope='bn1', decay=0.9,
            #                                 zero_debias_moving_mean=True, variables_collections=['bn_collections'])
            bn1 = tf.contrib.layers.batch_norm(conv1,
                                            decay=0.9, epsilon=1e-05,
                                            center=True, scale=True, renorm=False, updates_collections=None,
                                            is_training=is_training, scope='bn1')
            r1 = tf.nn.relu(bn1)
            
            w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME') # NDHWC format
            b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.bias_add(conv2, b2)
            # bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2', decay=0.9,
            #                                 zero_debias_moving_mean=True, variables_collections=['bn_collections'])
            bn2 = tf.contrib.layers.batch_norm(conv2,
                                            decay=0.9, epsilon=1e-05,
                                            center=True, scale=True, renorm=False, updates_collections=None,
                                            is_training=is_training, scope='bn2')
            r2 = tf.nn.relu(bn2)
            #if self.regularization:
            self.regularizers.append(tf.nn.l2_loss(w1))
            self.regularizers.append(tf.nn.l2_loss(w2))
            self.regularizers.append(tf.nn.l2_loss(b1))
            self.regularizers.append(tf.nn.l2_loss(b2))
            return r2

    def res_conv3d(self, input_, output_dim, f_size, is_training, scope='conv3d'):
        with tf.variable_scope(scope) as scope:
            # VGG network uses two 3*3 conv layers to effectively increase receptive field
            w1 = tf.get_variable('w1', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1)) 
                                
            conv1 = tf.nn.conv3d(input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME') # NDHWC format
            b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.bias_add(conv1, b1)
            # bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training, scope='bn1', decay=0.9,
            #                                 zero_debias_moving_mean=True, variables_collections=['bn_collections'])
            bn1 = tf.contrib.layers.batch_norm(conv1,
                                            decay=0.9, epsilon=1e-05,
                                            center=True, scale=True, renorm=False, updates_collections=None,
                                            is_training=is_training, scope='bn1')
            r1 = tf.nn.relu(bn1)
            
            w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME') # NDHWC format
            b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.bias_add(conv2, b2)
            # bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2', decay=0.9,
            #                                 zero_debias_moving_mean=True, variables_collections=['bn_collections'])
            bn2 = tf.contrib.layers.batch_norm(conv2,
                                            decay=0.9, epsilon=1e-05,
                                            center=True, scale=True, renorm=False, updates_collections=None,
                                            is_training=is_training, scope='bn2')
            r2 = tf.nn.relu(bn2)
            #if self.regularization:
            self.regularizers.append(tf.nn.l2_loss(w1))
            self.regularizers.append(tf.nn.l2_loss(w2))
            self.regularizers.append(tf.nn.l2_loss(b1))
            self.regularizers.append(tf.nn.l2_loss(b2))
            return r2+input_

    def deconv3d(self, input_, output_shape, f_size, is_training, scope='deconv3d'):
        with tf.variable_scope(scope) as scope:
            output_dim = output_shape[-1]
            w = tf.get_variable('w', [f_size, f_size, f_size, output_dim, input_.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            deconv = tf.nn.conv3d_transpose(input_, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')  # NDHWC format
            # bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn', decay=0.9,
            #                                 zero_debias_moving_mean=True, variables_collections=['bn_collections'])
            bn = tf.contrib.layers.batch_norm(deconv,
                                            decay=0.9, epsilon=1e-05,
                                            center=True, scale=True, renorm=False, updates_collections=None,
                                            is_training=is_training, scope='bn')
            r = tf.nn.relu(bn)
            return r
        
    def crop_and_concat(self, x1, x2):
        x1_shape = x1.get_shape().as_list()
        x2_shape = x2.get_shape().as_list()
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 4)

    def conv_relu(self, input_, output_dim, f_size, s_size, scope='conv_relu'):
        with tf.variable_scope(scope) as scope:
            w = tf.get_variable('w', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv = tf.nn.conv3d(input_, w, strides=[1, s_size, s_size, s_size, 1], padding='SAME')
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)
            r = tf.nn.relu(conv)
            return r

    def phase_shift(self, I):
       # Helper function with main phase shift operation
        bsize, width, height, depth, channel = I.get_shape().as_list()
    # #    X = tf.reshape(I, (self.batch_size, a, b, r, r))
    # #    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    # #    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    # #    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    # #    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
    #     X = tf.split(I, channel, axis = 3)  # z, [bsize, x, y, R]
    #     X = tf.concat([tf.squeeze(x) for x in X], axis=3)  # bsize, x, y, z*R
        X = tf.transpose(I, (0,3,1,2,4)) # bsize, z, x, y, r
        X = tf.split(X, depth, 1)  # z, [bsize, x, y, r]
        X = tf.concat([tf.squeeze(x) for x in X], 3)  # bsize, x, y, z*r
        print X.get_shape().as_list()

        return X

    # def batch_norm(self, x, phase_train, name = 'batchnorm', reuse = False):
    #     return tf.contrib.layers.batch_norm(x,
    #             decay=0.99,
    #             updates_collections=None,
    #             epsilon=1e-5,
    #             scale=True,
    #             is_training=phase_train,
    #             scope=name)

    # def add_bias(self, x):
    #     N, M, F = x.get_shape().as_list()
    #     # b = self._bias_variable([1, 1, int(F)], regularization=True)
    #     b = self._bias_variable([1, int(M), int(F)], regularization=True)
    #     return x + b

    # def _bias_variable(self, shape, regularization=True):
    #     initial = tf.constant_initializer(self._params["training_parameters"]["bias"]["value"])
    #     var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
    #     if regularization:
    #         self.regularizers.append(tf.nn.l2_loss(var))
    #         #self.regularizers.append(tf.reduce_sum(tf.abs(var)))
    #     # tf.summary.histogram(var.op.name, var)
    #     return var

    # def b1relu(self, x):
    #     """Bias and ReLU. One bias per filter."""
    #     return tf.nn.relu(x)

    # def tanh(self, x):
    #     return tf.nn.tanh(x)

    # def lrelu(self, x):
    #     return tf.maximum(x, self._params["training_parameters"]["alpha"]["value"]*x)

    # def _weight_variable(self, shape, name, regularization=True):
    #     # initial = tf.truncated_normal_initializer(self._params["training_parameters"]["weightMean"]["value"], self._params["training_parameters"]["weightStd"]["value"])
    #     # initial = tf.contrib.layers.xavier_initializer()
    #     initial = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
    #     var = tf.get_variable(name, shape, tf.float32, initializer=initial)
    #     if regularization:
    #         self.regularizers.append(tf.nn.l2_loss(var))
    #         #self.regularizers.append(tf.reduce_sum(tf.abs(var)))
    #         # self.regularizers.append(tf.losses.absolute_difference(var))
    #     # tf.summary.histogram(var.op.name, var)
    #     return var

