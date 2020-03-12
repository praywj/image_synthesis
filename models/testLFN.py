from models.load_data import *
from models.model import *
import time
import matplotlib.pyplot as plt


def train():
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    CAPACITY = 1000

    train_dir = 'D:\\edit\\contour1\\image_train'
    label_dir = 'D:\\edit\\Images\\image_train'
    rgb_dir = 'D:\\edit\\color1\\image_train'
    gradient_dir = 'D:\\edit\\gradient\\image_train'
    logs_dir = 'logs_2\\'
    max_step = 200000
    sess = tf.Session()

    train_list = get_train_list(train_dir, label_dir, rgb_dir, gradient_dir, is_random=True)
    image_train_batch, label_train_batch, rgb_train_batch, gradient_train_batch = \
        get_train_batch(train_list, [IMAGE_SIZE, IMAGE_SIZE], BATCH_SIZE, CAPACITY, is_random=True)


    input = get_inputs(image_train_batch,
                       rgb_train_batch,
                       gradient_train_batch)
    out_batch = generator(input, is_training=True, trainable=True, name='LFN')
    mse = tf.reduce_mean(tf.square(label_train_batch - out_batch))
    global_step = tf.train.get_or_create_global_step(sess.graph)
    g_lr = tf.train.exponential_decay(2e-4, global_step, 1e4, 0.98, staircase=True)
    gen_optim = tf.train.AdamOptimizer(g_lr).minimize(mse, global_step=global_step)

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
    #增加通道数 颜色梯度信息
    image_test = tf.concat([image_rgb, image_test], axis=3)
    image_test = tf.concat([image_gradient, image_test], axis=3)

    image_gray = tf.read_file('D:\\edit\\test\\image_242.png')
    image_gray = tf.image.decode_png(image_gray, channels=3)
    image_gray = tf.image.resize_images(image_gray, [256, 256])
    image_gray = tf.cast(image_gray, tf.float32) / 255.0
    image_gray = tf.reshape(image_gray, [1, 256, 256, 3])
    output = generator(image_test, is_training=False, name='LFN')

    output = tf.clip_by_value(output, 0, 1)
    test_psnr = 10 * tf.log(1 / (tf.reduce_mean(tf.square(output - image_gray)))) / np.log(10)

    # tf.summary.scalar('MSE', mse)
    # summary_op = tf.summary.merge_all()
    # file_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    start_time = time.time()
    try:
        for step in range(max_step):
            if coord.should_stop():
                break

            # img_train, img_label = sess.run([train_batch, real_batch])
            # img_train = img_train[0, :, :, 0]
            # img_label = img_label[0, :, :, 0]
            # plt.subplot(1, 2, 1), plt.imshow(img_train, 'gray')
            # plt.subplot(1, 2, 2), plt.imshow(img_label, 'gray')
            # plt.show()

            _ = sess.run(gen_optim)

            if step % 100 == 0:
                g_loss, lr, lena_psnr = sess.run([mse, g_lr, test_psnr])
                runtime = time.time() - start_time
                start_time = time.time()
                print('Step: %6d, mse: %.8f, learning_rate: %g, lena_psnr: %2.2fdB, runtime: %3.2fs'
                      % (step, g_loss, lr, lena_psnr, runtime))

            if step % 1000 == 0:
                img = sess.run(output)
                save_path = 'D:\\edit\\results\\train2\\step-{0}.bmp'.format(step)

                # img = Image.fromarray(img[0, :, :, 0] * 255)
                # img.convert('L').save(save_path)
                plt.imsave(save_path, img[0])
            if step % (max_step / 10) == 0 or step == max_step - 1:
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


def eval():
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    CAPACITY = 1000

    train_dir = 'D:\\edit\\contour\\image_test'
    label_dir = 'D:\\edit\\Images\\image_test'
    rgb_dir = 'D:\\edit\\color\\image_test'
    gradient_dir = 'D:\\edit\\gradient\\image_test'
    logs_dir = 'logs_2\\'

    sess = tf.Session()

    train_list = get_train_list(train_dir, label_dir, rgb_dir, gradient_dir, is_random=True)
    image_train_batch, label_train_batch, rgb_train_batch, gradient_train_batch = \
        get_train_batch(train_list, [IMAGE_SIZE, IMAGE_SIZE], BATCH_SIZE, CAPACITY, is_random=True)

    input = get_inputs(image_train_batch, rgb_train_batch, gradient_train_batch)
    out_batch = generator(input, is_training=True, trainable=True, name='LFN')
    out_batch = tf.clip_by_value(out_batch, 0, 0.99)

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
        for step in range(len(train_list[0])):
            if coord.should_stop():
                break

            lena = sess.run(out_batch)
            filename = train_list[0][step].split('\\')[-1]
            plt.imsave('D:\\edit\\results\\LFN_test\\' + filename, lena[0])

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)

    sess.close()


if __name__ == '__main__':
    train()
    #eval()