from models.ops import *


def generator(inputs, is_training=True, trainable=True, name='None'):

    print('==========', name, '==========')
    ngf = 64
    batch_norm = True
    kernel_size = 4
    print('Inputs:', inputs)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = conv2d(inputs, ngf, kernel_size, 2, activation_fn='leakyrelu', batch_norm=False,
                       is_training=is_training, trainable=trainable, name='conv1')
        print(conv1)
        conv2 = conv2d(conv1, ngf * 2, kernel_size, 2, activation_fn='leakyrelu', batch_norm=False,
                       is_training=is_training, trainable=trainable, name='conv2')
        print(conv2)
        conv3 = conv2d(conv2, ngf * 4, kernel_size, 2, activation_fn='leakyrelu', batch_norm=False,
                       is_training=is_training, trainable=trainable, name='conv3')
        print(conv3)
        conv4 = conv2d(conv3, ngf * 8, kernel_size, 2, activation_fn='leakyrelu', batch_norm=False,
                       is_training=is_training, trainable=trainable, name='conv4')
        print(conv4)
        conv5 = conv2d(conv4, ngf * 8, kernel_size, 2, activation_fn='leakyrelu', batch_norm=False,
                       is_training=is_training, trainable=trainable, name='conv5')
        print(conv5)

        mid = conv2d(conv5, ngf * 8, kernel_size, 2, activation_fn='leakyrelu', batch_norm=False,
                     is_training=is_training, trainable=trainable, name='mid')
        print(mid)
        mid = batchnorm(mid)
        deconv1_ = conv2d_transpose(mid, ngf * 8, kernel_size, 2, activation_fn='relu', batch_norm=batch_norm,
                                    is_training=is_training, trainable=trainable, name='deconv1')
        deconv1 = tf.concat([deconv1_, conv5], axis=3)
        print(deconv1)
        deconv2_ = conv2d_transpose(deconv1, ngf * 8, kernel_size, 2, activation_fn='relu', batch_norm=batch_norm,
                                    is_training=is_training, trainable=trainable, name='deconv2')
        deconv2 = tf.concat([deconv2_, conv4], axis=3)
        print(deconv2)
        #deconv2 = batch_norm(deconv2)
        deconv3_ = conv2d_transpose(deconv2, ngf * 4, kernel_size, 2, activation_fn='relu', batch_norm=batch_norm,
                                    is_training=is_training, trainable=trainable, name='deconv3')
        deconv3 = tf.concat([deconv3_, conv3], axis=3)
        print(deconv3)
        deconv4_ = conv2d_transpose(deconv3, ngf * 2, kernel_size, 2, activation_fn='relu', batch_norm=batch_norm,
                                    is_training=is_training, trainable=trainable, name='deconv4')
        deconv4 = tf.concat([deconv4_, conv2], axis=3)
        print(deconv4)
        deconv5_ = conv2d_transpose(deconv4, ngf, kernel_size, 2, activation_fn='relu', batch_norm=batch_norm,
                                    is_training=is_training, trainable=trainable, name='deconv5')
        deconv5 = tf.concat([deconv5_, conv1], axis=3)
        print(deconv5)
        deconv6 = conv2d_transpose(deconv5, ngf, kernel_size, 2, activation_fn='relu', batch_norm=batch_norm,
                                   is_training=is_training, trainable=trainable, name='deconv6')
        print(deconv6)

        output = conv2d(deconv6, 3, kernel_size, 1, activation_fn='relu', batch_norm=False,
                        is_training=is_training, trainable=trainable, name='output')
        #output = batchnorm(output)
        print(output, end='\n\n')

    return output


def discriminator(inputs, label, is_training=True, trainable=True, name='None'):

    print('==========', name, '==========')
    ngf = 64
    batch_norm = False
    kernel_size = 4
    inputs = tf.concat([inputs, label], axis=3)
    print('Inputs:', inputs)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = conv2d(inputs, ngf, kernel_size, 2, activation_fn='leakyrelu', batch_norm=batch_norm,
                       is_training=is_training, trainable=trainable, name='conv1')
        print(conv1)
        conv2 = conv2d(conv1, ngf * 2, kernel_size, 2, activation_fn='leakyrelu', batch_norm=batch_norm,
                       is_training=is_training, trainable=trainable, name='conv2')
        print(conv2)
        conv3 = conv2d(conv2, ngf * 4, kernel_size, 2, activation_fn='leakyrelu', batch_norm=batch_norm,
                       is_training=is_training, trainable=trainable, name='conv3')
        print(conv3)

        dilated1 = dilated2d(conv3, ngf * 4, kernel_size,  'leakyrelu', 2, batch_norm, is_training, trainable, 'dilated1')
        dilated2 = dilated2d(conv3, ngf * 4, kernel_size,  'leakyrelu', 4, batch_norm, is_training, trainable, 'dilated2')
        dilated3 = dilated2d(conv3, ngf * 4, kernel_size,  'leakyrelu', 8, batch_norm, is_training, trainable, 'dilated3')
        dilated4 = dilated2d(conv3, ngf * 4, kernel_size,  'leakyrelu', 12, batch_norm, is_training, trainable, 'dilated4')
        dilated = tf.concat([dilated1, dilated2, dilated3, dilated4], axis=3)
        print(dilated)

        logits = conv2d(dilated, 1, kernel_size, 1, 'leakyrelu', False, is_training, trainable, 'logits')
        print(logits, end='\n\n')
        out_prob = tf.nn.softmax(logits)

    return logits,out_prob


def get_inputs(contours, image_rgb, gradient):
    input1 = tf.concat([image_rgb, contours], axis=3)
    output1 = tf.concat([gradient, input1], axis=3)

    return output1


def build_model(contours, grounds, global_step1, image_rgb, gradient):
    assert contours.get_shape()[-1] == 1, 'Contours images''s channel must be 1.'
    assert grounds.get_shape()[-1] == 3, 'Ground truth images''s channel must be 3'
    assert image_rgb.get_shape()[-1] == 3
    assert gradient.get_shape()[-1] == 1

    input = get_inputs(contours, image_rgb, gradient)

    lf_image = generator(input, is_training=False, trainable=False, name='LFN')

    inputs = tf.concat([lf_image, input], axis=3)
    """Loss函数"""
    fake_image = generator(inputs, is_training=True, trainable=True, name='HFN')

    logits_fake,logits_fake_prob = discriminator(fake_image, contours, is_training=True, trainable=True, name='Discrim')
    logits_real,logits_real_prob = discriminator(grounds, contours, is_training=True, trainable=True, name='Discrim')

    D_loss_fake = tf.nn.l2_loss(logits_fake - tf.zeros_like(logits_fake))  #Computes half the L2 norm of a tensor without the sqrt
    D_loss_real = tf.nn.l2_loss(logits_real - tf.zeros_like(logits_real))
    # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
    #                                                                      labels=tf.zeros_like(logits_fake)))
    # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,
    #                                                                      labels=tf.ones_like(logits_real)))
    D_loss = D_loss_fake + D_loss_real

    G_loss_advs = tf.nn.l2_loss(logits_fake - tf.ones_like(logits_fake))
    # G_loss_l1 = tf.reduce_mean(tf.abs(fake_image - grounds))
    # G_loss_advs = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
    #                                                                      labels=tf.ones_like(logits_fake)))
    G_loss_l1 = tf.reduce_mean(tf.square(fake_image - grounds))
    G_loss = G_loss_advs + G_loss_l1

    vars = tf.trainable_variables()
    d_vars = [var for var in vars if 'Discrim' in var.name]
    g_vars = [var for var in vars if 'HFN' in var.name]

    lr_d = tf.train.exponential_decay(2e-5, global_step1, 1e4, 0.98, staircase=True)
    lr_g = tf.train.exponential_decay(2e-4, global_step1, 1e4, 0.98, staircase=True)
    D_optim = tf.train.AdamOptimizer(lr_d).minimize(D_loss, var_list=d_vars, global_step=global_step1)
    G_optim = tf.train.AdamOptimizer(lr_g).minimize(G_loss, var_list=g_vars, global_step=global_step1)

    """Summary"""
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('L1_loss', G_loss_l1)
    return D_loss, G_loss, G_loss_l1, D_optim, G_optim, logits_real_prob, logits_fake_prob,lr_d,lr_g
    # return D_optim, G_optim, D_loss, G_loss, G_loss_l1, lr_d, lr_g, logits_real_prob, logits_fake_prob


if __name__ == '__main__':
    inputs = tf.ones([16, 256, 256, 1])
    inputs2 = tf.ones([16, 256, 256, 3])
    sess = tf.Session()
    global_step = tf.train.get_or_create_global_step(sess.graph)
    build_model(inputs, inputs2, global_step)