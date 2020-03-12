from models.load_data import *
from models.model import *
import time
import matplotlib.pyplot as plt
import tensorflow as tf

def train():
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    CAPACITY = 1000

    train_dir = 'D:\\edit\\contour\\image_train'
    label_dir = 'D:\\edit\\Images\\image_train'
    rgb_dir = 'D:\\edit\\color\\image_train'
    gradient_dir = 'D:\\edit\\gradient\\image_train'
    logs_dir = 'logs_3\\'
    max_step = 200000
    sess = tf.Session()

    train_list = get_train_list(train_dir, label_dir, rgb_dir, gradient_dir, is_random=True)
    image_train_batch, label_train_batch, rgb_train_batch, gradient_train_batch = \
        get_train_batch(train_list, [IMAGE_SIZE, IMAGE_SIZE], BATCH_SIZE, CAPACITY, is_random=True)
    # Returns and create (if necessary) the global step tensor.
    global_steps = tf.train.get_or_create_global_step(sess.graph)
    d_loss, g_loss, mse_loss, d_optic, g_optic, d_real_p, d_fake_p, lrd, lrg = build_model(image_train_batch,
                                                                                           label_train_batch,
                                                                                           global_steps,
                                                                                           rgb_train_batch,
                                                                                           gradient_train_batch)

    # out_batch = generator(image_train_batch, is_training=True, name='LFN')
    #
    #
    # global_step = tf.train.get_or_create_global_step()
    # lr = tf.train.exponential_decay(1e-4, global_step, 2e4, 1, staircase=True)
    # G_optim = tf.train.AdamOptimizer(lr).minimize(mse, global_step=global_step)

    image_test = tf.read_file('D:\\edit\\test\\contour_242.png')
    image_test = tf.image.decode_png(image_test, channels=1)
    image_test = tf.image.resize_images(image_test, [256, 256])
    image_test = tf.cast(image_test, tf.float32) / 255.0
    image_test = tf.reshape(image_test, [1, 256, 256, 1])

    image_rgb = tf.read_file('D:\\edit\\test\\color_242.png')
    image_rgb = tf.image.decode_png(image_rgb, channels=3)
    image_rgb = tf.image.resize_images(image_rgb, [256, 256])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.0
    image_rgb = tf.reshape(image_rgb, [1, 256, 256, 3])

    image_gradient = tf.read_file('D:\\edit\\test\\gradient_242.png')
    image_gradient = tf.image.decode_png(image_gradient, channels=1)
    image_gradient = tf.image.resize_images(image_gradient, [256, 256])
    image_gradient = tf.cast(image_gradient, tf.float32) / 255.0
    image_gradient = tf.reshape(image_gradient, [1, 256, 256, 1])
    """增加通道数 颜色梯度信息"""
    image_test = tf.concat([image_rgb, image_test], axis=3)
    image_test = tf.concat([image_gradient, image_test], axis=3)

    image_gray = tf.read_file('D:\\edit\\test\\image_242.png')
    image_gray = tf.image.decode_png(image_gray, channels=3)
    image_gray = tf.image.resize_images(image_gray, [256, 256])
    image_gray = tf.cast(image_gray, tf.float32) / 255.0
    image_gray = tf.reshape(image_gray, [1, 256, 256, 3])

    LFNoutput = generator(image_test, is_training=False, name='LFN')
    mse = tf.reduce_mean(tf.square(label_train_batch - LFNoutput))
    inputs = tf.concat([LFNoutput, image_test], axis=3)
    image_test = generator(inputs, is_training=True, trainable=True, name='HFN')
    image_test = tf.clip_by_value(image_test, 0, 1)
    test_psnr = 10 * tf.log(1 / (tf.reduce_mean(tf.square(image_test - image_gray)))) / np.log(10)
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('MSE', mse)
    tf.summary.scalar('D_loss', d_loss)
    tf.summary.scalar('G_loss', g_loss)
    tf.summary.scalar('L1_loss', mse_loss)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    start_time = time.time()
    k = 2
    try:
        for step in range(max_step):
            if coord.should_stop():
                break

            # img_train, img_noise, img_label = sess.run([train_batch, noise_batch, label_batch])
            # img_train = img_train[0, :, :, 0]
            # img_noise = img_noise[0, :, :, 0]
            # img_label = img_label[0, :, :, 0]
            # plt.subplot(1, 3, 1), plt.imshow(img_train, 'gray')
            # plt.subplot(1, 3, 2), plt.imshow(img_noise, 'gray')
            # plt.subplot(1, 3, 3), plt.imshow(img_label, 'gray')
            # plt.show()
            if step % k == 0:
                _ = sess.run([d_optic])
            else:
                _ = sess.run([g_optic])

            loss_d, loss_g, loss_mse, lr_d = sess.run([d_loss, g_loss, mse_loss, lrd])

            if step % 100 == 0:
                lena_psnr, p_real, p_fake = sess.run([test_psnr, d_real_p, d_fake_p])
                print('Step: %d, d_loss: %.8f, g_loss: %.8f,  g_loss_l1:%.8f, lr_discrim: %g, '
                      'time:%.2fs, lena_psnr:%.2fdB'
                      % (step, loss_d, loss_g,  loss_mse, lr_d, time.time() - start_time, lena_psnr))
                # print('d_real_prob = %.2f%%, d_fake_p = %.2f%%' % (p_real*100, p_fake*100))
                start_time = time.time()
                result = sess.run(summary_op)
                writer.add_summary(result, step)
            if step % 1000 == 0:
                img = sess.run(image_test)
                save_path = 'D:\\edit\\results\\train\\step-{0}.bmp'.format(step)

                # img = Image.fromarray(img[0, :, :, 0] * 255)
                # img.convert('L').save(save_path)
                plt.imsave(save_path, img[0])
            if step % (max_step / 10) == 0 or step == max_step - 1:
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                # lena, lena_psnr = sess.run([image_test, test_psnr])
                # print("\n===== PSNR:%.2f =====\n" % lena_psnr)
                # plt.imshow(lena[0, :, :, 0], 'gray')
                # plt.show()
                # img = Image.fromarray(lena[0, :, :, 0] * 255)
                # img.convert('RGB').save('vis\\' + str(step) + '.bmp')
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


def eval():

    logs_dir = 'F:\\code\\EditImagesUseContours\\models\\logs_3\\'
    sess = tf.Session()
    image_test = tf.read_file('F:\\test\\data\\contour.png')
    image_test = tf.image.decode_png(image_test, channels=1)
    image_test = tf.image.resize_images(image_test, [256, 256])
    image_test = tf.cast(image_test, tf.float32) / 255.0
    image_test = tf.reshape(image_test, [1, 256, 256, 1])

    image_rgb = tf.read_file('F:\\test\\data\\color.png')
    image_rgb = tf.image.decode_png(image_rgb, channels=3)
    image_rgb = tf.image.resize_images(image_rgb, [256, 256])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.0
    image_rgb = tf.reshape(image_rgb, [1, 256, 256, 3])

    image_gradient = tf.read_file('F:\\test\\data\\gradient.png')
    image_gradient = tf.image.decode_png(image_gradient, channels=1)
    image_gradient = tf.image.resize_images(image_gradient, [256, 256])
    image_gradient = tf.cast(image_gradient, tf.float32) / 255.0
    image_gradient = tf.reshape(image_gradient, [1, 256, 256, 1])
    """增加通道数 颜色梯度信息"""
    image_test = tf.concat([image_rgb, image_test], axis=3)
    image_test = tf.concat([image_gradient, image_test], axis=3)

    image_gray = tf.read_file('F:\\test\\image.png')
    image_gray = tf.image.decode_png(image_gray, channels=3)
    image_gray = tf.image.resize_images(image_gray, [256, 256])
    image_gray = tf.cast(image_gray, tf.float32) / 255.0
    image_gray = tf.reshape(image_gray, [1, 256, 256, 3])

    image_test = generator(image_test, is_training=False, name='LFN')
    # inputs = tf.concat([LFNoutput, image_test], axis=3)
    # image_test = generator(inputs, is_training=True, trainable=True, name='HFN')
    out_batch = tf.clip_by_value(image_test, 0, 1)

    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        lena = sess.run(out_batch)
        plt.imsave('F:\\test\\data\\result.png',lena[0])

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)

    sess.close()


if __name__ == '__main__':
     #train()
     eval()
