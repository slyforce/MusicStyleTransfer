import tensorflow as tf
import numpy as np


def conv_layer(
    x, scope, initializer=None, filters=64, stride=(
        1, 1), kernel_size=(
            3, 3), name=''):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(
            x,
            kernel_initializer=initializer,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            name=name)
    return x


def batch_norm(scope, x, is_training=True, name=''):
    # return x

    with tf.variable_scope(scope):
        x = tf.layers.batch_normalization(x, training=is_training, name=name)
    return x


def get_nr_variables(vars):
    tot_nb_params = 0
    for v in vars:
        shape = v.get_shape()  # e.g [D,F] or [W,H,C]
        params = 1
        for dim in range(0, len(shape)):
            params *= int(shape[dim])

        tot_nb_params += params

    return tot_nb_params


def deconv2d(input_, output_shape,
             variable_scope,
             height, width,
             stride1, stride2,
             name=''):

    with tf.variable_scope(variable_scope) as vs:
        relu_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(name + 'w',
                            [height,
                             width,
                             output_shape[-1],
                                input_.shape[-1]],
                            initializer=relu_initializer)

        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape, strides=[
                1, stride1, stride2, 1], name=name)

        biases = tf.get_variable(name + 'biases',
                                 [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv


def visualize_cnn_kernel(kernel_name, tensorboard_name):
    kernel = [
        v for v in tf.global_variables() if str(
            v.name).startswith(kernel_name)][0]
    # shape  = kernelsize x in_channel x out_channel
    kernel = tf.transpose(kernel, [2, 3, 0, 1])
    kernel = tf.reshape(
        kernel, [
            kernel.shape[0] * kernel.shape[1], kernel.shape[2], kernel.shape[3]])
    kernel = tf.expand_dims(kernel, axis=3)
    tf.summary.image(tensorboard_name, kernel, max_outputs=100)


def MIDINetDiscriminator(x, feature_size, is_training=True, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse) as vs:
        relu_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)

        x = conv_layer(
            x, vs, filters=14, stride=(
                2, 2), kernel_size=(
                2, feature_size), name='d-conv-1')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        #x = batch_norm(vs, x, is_training, 'd-batch-norm-1')

        x = conv_layer(
            x, vs, filters=24, stride=(
                2, 2), kernel_size=(
                2, 1), name='d-conv-2')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        #x = batch_norm(vs, x, is_training, 'd-batch-norm-2')

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 128, activation=None, name='d-dense-1')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        #x = batch_norm(vs, x, is_training, 'd-batch-norm-3')

        x = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='d-output')

    variables = tf.contrib.framework.get_variables(vs)
    print "Discriminator variables:", get_nr_variables(variables)

    if not reuse:
        visualize_cnn_kernel('Discriminator/d-conv-1', 'discriminator_kernel1')
        visualize_cnn_kernel('Discriminator/d-conv-2', 'discriminator_kernel2')

    return x, variables


def MIDINetGenerator(
        batch_size,
        sequence_length,
        feature_size,
        reuse=False,
        is_training=True):
    # glorot initialization
    relu_initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)

    with tf.variable_scope("Generator", reuse=reuse) as vs:
        x = tf.random_normal(mean=0., stddev=1.0, shape=[batch_size, 100],
                             name='g_noise')  # random gaussian sample of z_dim

        #x = tf.layers.dense(x, 1024, kernel_initializer=relu_initializer, name='dense-1')
        #x = tf.nn.leaky_relu(x, alpha=0.2)

        x = tf.layers.dense(x,
                            int(np.ceil(sequence_length / 8.)) * int(np.ceil(feature_size / 8.)),
                            kernel_initializer=relu_initializer,
                            name='g-dense-2')
        x = tf.nn.leaky_relu(x, alpha=0.2)

        print "shape of noise projection", x.shape

        x = tf.reshape(x,
                       shape=[batch_size,
                              int(np.ceil(sequence_length / 8.)),
                              int(np.ceil(feature_size / 8.)),
                              1])

        print "shape of initial image", x.shape

        x = deconv2d(x,
                     [batch_size,
                      int(np.ceil(sequence_length / 4.)),
                         int(np.ceil(feature_size / 4.)),
                         128],
                     vs,
                     2,
                     1,
                     2,
                     2,
                     'g-deconv-1')
        #x = batch_norm(vs, x, is_training, 'g-batch-norm-1')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print "after deconv1", x.shape

        x = deconv2d(x,
                     [batch_size,
                      int(np.ceil(sequence_length / 2)),
                         int(np.ceil(feature_size / 2.)),
                         128],
                     vs,
                     2,
                     1,
                     2,
                     2,
                     'g-deconv-2')
        #x = batch_norm(vs, x, is_training, 'g-batch-norm-2')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print "after deconv2", x.shape

        x = deconv2d(x, [batch_size, sequence_length,
                         feature_size, 128], vs, 2, 1, 2, 2, 'g-deconv-3')
        #x = batch_norm(vs, x, is_training, 'g-batch-norm-3')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print "after deconv3", x.shape

        x = deconv2d(x,
                     [batch_size,
                      sequence_length,
                      feature_size,
                      1],
                     vs,
                     2,
                     feature_size,
                     1,
                     1,
                     'g-deconv-4')
        print "after deconv4", x.shape

        x = tf.nn.softmax(x * 100, axis=2)
        #x = tf.nn.sigmoid(x)
        #x = tf.nn.tanh(x)

        if not reuse:
            tf.summary.image('generator_image_final', tf.transpose(
                x, [0, 2, 1, 3]), max_outputs=1)
            visualize_cnn_kernel('Generator/g-deconv-1w', 'generator_kernel1')
            visualize_cnn_kernel('Generator/g-deconv-2w', 'generator_kernel2')
            visualize_cnn_kernel('Generator/g-deconv-3w', 'generator_kernel3')
            visualize_cnn_kernel('Generator/g-deconv-4w', 'generator_kernel4')

    variables = tf.contrib.framework.get_variables(vs)
    print "Generator variables:", get_nr_variables(variables)

    return x, variables


def MIDINetGeneratorFFNN(
        batch_size,
        sequence_length,
        feature_size,
        is_training=False,
        reuse=False):
    with tf.variable_scope("Generator", reuse=reuse) as vs:
        x = tf.random_normal(
            mean=0.,
            stddev=1.0,
            shape=[
                batch_size,
                100],
            name='g_noise')  # random gaussian sample of z_dim

        x = tf.layers.dense(x, 1024, name='dense-1')
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = tf.layers.dense(x, 1024, name='dense-2')
        x = tf.nn.leaky_relu(x, alpha=0.2)

        #x = tf.layers.dense(x, 256, name='dense-3')
        #x = tf.nn.leaky_relu(x, alpha=0.2)

        x = tf.layers.dense(
            x,
            sequence_length *
            feature_size,
            activation=None,
            name='g_output_layer')
        x = tf.reshape(x, [batch_size, sequence_length, feature_size, 1])

        x = tf.nn.softmax(x, axis=2)

        if not reuse:
            tf.summary.image('generator_image', tf.transpose(x, [0, 2, 1, 3]))

    variables = tf.contrib.framework.get_variables(vs)
    print "Generator variables:", get_nr_variables(variables)

    return x, variables


def MIDINetGeneratorSampleInput(
        input,
        input_classes,
        batch_size,
        sequence_length,
        feature_size,
        reuse=False,
        is_training=True):
    # input has a format of [B,H,W,Ch]
    with tf.variable_scope("Generator", reuse=reuse) as vs:

        # encoding of the conditioning sequence
        h1 = tf.layers.conv2d(
            input, kernel_size=(
                2, 1), strides=(
                2, 2), filters=10, name='input-conv-1', padding='SAME')
        h1 = tf.nn.leaky_relu(h1, alpha=0.2)

        h2 = tf.layers.conv2d(
            h1, kernel_size=(
                2, 1), strides=(
                2, 2), filters=10, name='input-conv-2', padding='SAME')
        h2 = tf.nn.leaky_relu(h2, alpha=0.2)

        h3 = tf.layers.conv2d(
            h2, kernel_size=(
                2, 1), strides=(
                2, 2), filters=10, name='input-conv-3', padding='SAME')
        h3 = tf.nn.leaky_relu(h3, alpha=0.2)

        # input noise
        x = tf.random_normal(mean=0., stddev=1.0, shape=[batch_size, 100],
                             name='g_noise')

        # feed-forward noise projection
        x = tf.layers.dense(x, int(np.ceil(sequence_length / 8.)) * int(
            np.ceil(feature_size / 8.)), kernel_initializer=None, name='g-dense-2')
        x = tf.nn.leaky_relu(x, alpha=0.2)

        # re-shape into 2d image with 1 channel
        x = tf.reshape(x,
                       shape=[batch_size,
                              int(np.ceil(sequence_length / 8.)),
                              int(np.ceil(feature_size / 8.)),
                              1])

        x = tf.concat([x, h3], axis=3)
        x = deconv2d(x,
                     [batch_size,
                      int(np.ceil(sequence_length / 4.)),
                         int(np.ceil(feature_size / 4.)),
                         128],
                     vs,
                     2,
                     1,
                     2,
                     2,
                     'g-deconv-1')
        #x = batch_norm(vs, x, is_training, 'g-batch-norm-1')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print "after deconv1", x.shape

        x = tf.concat([x, h2], axis=3)
        x = deconv2d(x,
                     [batch_size,
                      int(np.ceil(sequence_length / 2)),
                         int(np.ceil(feature_size / 2.)),
                         128],
                     vs,
                     2,
                     1,
                     2,
                     2,
                     'g-deconv-2')
        #x = batch_norm(vs, x, is_training, 'g-batch-norm-2')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print "after deconv2", x.shape

        x = tf.concat([x, h1], axis=3)
        x = deconv2d(x, [batch_size, sequence_length,
                         feature_size, 128], vs, 2, 1, 2, 2, 'g-deconv-3')
        #x = batch_norm(vs, x, is_training, 'g-batch-norm-3')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print "after deconv3", x.shape

        x = tf.concat([x, input], axis=3)
        x = deconv2d(x,
                     [batch_size,
                      sequence_length,
                      feature_size,
                      1],
                     vs,
                     2,
                     feature_size,
                     1,
                     1,
                     'g-deconv-4')
        print "after deconv4", x.shape

        x = tf.nn.softmax(x * 100, axis=2)

        if not reuse:
            tf.summary.image('generator_image', tf.transpose(x, [0, 2, 1, 3]))

    variables = tf.contrib.framework.get_variables(vs)
    print "Generator variables:", get_nr_variables(variables)

    return x, variables


def column_distance_loss(x):
    '''
    Computes the distance of neighbouring columns of the image
    :param x: Tensor of shape [B,H,W,Ch]
    :return: loss
    '''

    # pad the left and right column with 0 values
    x = tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]],
               "CONSTANT", constant_values=0)

    # take all columns except the last one
    # and subtract all columns except the first one
    # this is equal to subtracting all neighbouring columns in an iterative
    # manner
    x = (x[:, 0:-2, :, :] - x[:, 1:-1, :, :]) ** 2

    # reduce over all dimensions
    loss = tf.reduce_mean(x, axis=3)
    loss = tf.reduce_mean(loss, axis=2)
    loss = tf.reduce_mean(loss, axis=1)

    return loss
