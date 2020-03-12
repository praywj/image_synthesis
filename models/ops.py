import tensorflow as tf
import tensorflow.contrib.layers as layers


# convolution layer
def conv2d(inputs, output_dim, kernel_size, stride, activation_fn,  batch_norm=False,
           is_training=True, trainable=True, name=None):
    """
    Custom convolution_2d operation.
    :param inputs: a tensor with shape of [N, H, W, C]
    :param output_dim: a int, output feature dimensions
    :param kernel_size: a int, filter size
    :param stride: a int
    :param activation_fn: a string
    :param batch_norm:
    :param is_training: indicate this layer is used for training or testing
    :param trainable：
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        # convolution
        weights = tf.get_variable(name='w',
                                  shape=[kernel_size, kernel_size, inputs.get_shape()[-1].value, output_dim],
                                  dtype=tf.float32,
                                  trainable=trainable,
                                  initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(input=inputs,
                            filter=weights,
                            strides=[1, stride, stride, 1],
                            padding='SAME')

        # batch normalization
        if batch_norm is True:
            conv = layers.batch_norm(inputs=conv,
                                     updates_collections=None,
                                     trainable=trainable,
                                     is_training=is_training)
            # conv = group_norm(inputs=conv,
            #                   G=32,
            #                   eps=1e-5)
        else:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     trainable=trainable,
                                     initializer=tf.zeros_initializer())
            # conv = tf.nn.bias_add(conv, biases)
            conv = conv + biases

        # activation function
        if activation_fn.lower() == 'relu':
            return tf.nn.relu(conv)
        elif activation_fn.lower() == 'leakyrelu':
            return tf.nn.leaky_relu(conv, alpha=0.2)
        elif activation_fn.lower() == 'sigmoid':
            return tf.nn.sigmoid(conv)
        elif activation_fn.lower() == 'tanh':
            return tf.nn.tanh(conv)
        elif activation_fn.lower() == 'none':
            return conv
        else:
            return None
#定义批量归一化
def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
# 空洞卷积层
def dilated2d(inputs, output_dim, kernel_size, activation_fn, rate, batch_norm=False, is_training=True,
              trainable=True, name='dilated2d'):
    with tf.variable_scope(name):
        weights = tf.get_variable(name='w',
                                  shape=[kernel_size, kernel_size, inputs.get_shape()[-1], output_dim],
                                  dtype=tf.float32,
                                  # initializer=tf.truncated_normal_initializer(stddev=stddev)
                                  initializer=layers.xavier_initializer())
        dilated = tf.nn.atrous_conv2d(inputs, weights, rate, padding='SAME')

        # batch normalization
        if batch_norm is True:
            dilated = layers.batch_norm(inputs=dilated,
                                       updates_collections=None,
                                       trainable=trainable,
                                       is_training=is_training)
        else:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     trainable=trainable,
                                     initializer=tf.zeros_initializer())
            dilated = dilated + biases
        if activation_fn.lower() == 'relu':
            out = tf.nn.relu(dilated)
        elif activation_fn.lower() == 'leakyrelu':
            out = tf.nn.leaky_relu(dilated, alpha=0.2)
        elif activation_fn.lower() == 'sigmoid':
            out = tf.nn.sigmoid(dilated)
        elif activation_fn.lower() == 'tanh':
            out = tf.nn.tanh(dilated)
        elif activation_fn.lower() == 'none':
            out = dilated
        else:
            out = None
    return dilated

# convolution transpose layer
def conv2d_transpose(inputs, output_dim, kernel_size, stride, activation_fn, batch_norm=False, is_training=True, trainable=True, name=None):
    """
    Custom transpose convolution_2d operation.
    :param inputs: a tensor with shape of [N, H, W, C]
    :param output_dim: a int, output feature dimensions
    :param kernel_size: a int, filter size
    :param stride: a int
    :param activation_fn: a string
    :param batch_norm:
    :param is_training: indicate this layer is used for training or testing
    :param trainable:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        # deconvolution
        weights = tf.get_variable(name='w',
                                  shape=[kernel_size, kernel_size, output_dim, inputs.get_shape()[-1].value],
                                  dtype=tf.float32,
                                  trainable=trainable,
                                  initializer=layers.xavier_initializer())
        output_shape = inputs.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = output_dim
        deconv = tf.nn.conv2d_transpose(value=inputs,
                                        filter=weights,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding='SAME')

        # batch normalization
        if batch_norm is True:
            deconv = layers.batch_norm(inputs=deconv,
                                       updates_collections=None,
                                       trainable=trainable,
                                       is_training=is_training)
            # deconv = group_norm(inputs=deconv,
            #                     G=32,
            #                     eps=1e-5)
        else:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     trainable=trainable,
                                     initializer=tf.zeros_initializer())
            # deconv = tf.nn.bias_add(deconv, biases)
            deconv = deconv + biases

        # activation function
        if activation_fn.lower() == 'relu':
            return tf.nn.relu(deconv)
        elif activation_fn.lower() == 'leakyrelu':
            return tf.nn.leaky_relu(deconv, alpha=0.2)
        elif activation_fn.lower() == 'sigmoid':
            return tf.nn.sigmoid(deconv)
        elif activation_fn.lower() == 'tanh':
            return tf.nn.tanh(deconv)
        elif activation_fn.lower() == 'none':
            return deconv
        else:
            return None


# pooling layer
def avg_pool2d(inputs, kernel_size, stride, name=None):
    return tf.nn.avg_pool(inputs, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding='SAME', name=name)


# fully connected layer
def fully_connected(inputs, output_dim, activation_fn, batch_norm=False, is_training=True, trainable=True, name=None):
    with tf.variable_scope(name):
        weights = tf.get_variable(name='w',
                                  shape=[inputs.get_shape()[-1].value, output_dim],
                                  dtype=tf.float32,
                                  trainable=trainable,
                                  initializer=layers.xavier_initializer())
        fc = tf.matmul(inputs, weights)

        # batch normalization
        if batch_norm is True:
            fc = layers.batch_norm(inputs=fc,
                                   updates_collections=None,
                                   trainable=trainable,
                                   is_training=is_training)
        else:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     trainable=trainable,
                                     initializer=tf.zeros_initializer())
            fc = fc + biases

        # activation function
        if activation_fn.lower() == 'relu':
            return tf.nn.relu(fc)
        elif activation_fn.lower() == 'leakyrelu':
            return tf.nn.leaky_relu(fc, alpha=0.2)
        elif activation_fn.lower() == 'sigmoid':
            return tf.nn.sigmoid(fc)
        elif activation_fn.lower() == 'tanh':
            return tf.nn.tanh(fc)
        elif activation_fn.lower() == 'none':
            return fc
        else:
            return None


# group normalization
def group_norm(inputs, G=32, eps=1e-5, name='GroupNorm'):
    with tf.variable_scope(name):
        N, H, W, C = inputs.shape
        gamma = tf.get_variable(name='gamma',
                                shape=[1, 1, 1, C],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1))
        beta = tf.get_variable(name='beta',
                               shape=[1, 1, 1, C],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())

        inputs = tf.reshape(inputs, [N, G, H, W, C // G])
        mean, var = tf.nn.moments(inputs, [2, 3, 4], keep_dims=True)
        inputs = (inputs - mean) / tf.sqrt(var + eps)
        x = tf.reshape(inputs, [N, H, W, C])

    return x * gamma + beta


if __name__ == '__main__':
    inputs = tf.ones([50, 32, 32, 64])
    conv1 = conv2d(inputs, 128, 3, 1, 'relu', batch_norm=True, is_training=True, trainable=True, name='conv1')
    print(conv1)

    vars = tf.trainable_variables()
    for var in vars:
        print(var)