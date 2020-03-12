import tensorflow as tf
import numpy as np
import os
from PIL import Image


def get_all_files(file_root):
    """
    Traverse all file items in root dir recursively.
    :param file_root: path of files needed to traverse
    :return: a ndarray of paths of file items that in 'file_path'
    """
    filename_list = []

    for item in os.listdir(file_root):
        path = file_root + '\\' + item
        if os.path.isdir(path):     # if is a directory
            filename_list.extend(get_all_files(path))
        elif os.path.isfile(path):  # if is a file item
            filename_list.append(path)

    filename_list = np.asarray(filename_list)

    return filename_list


def get_train_list(train_dir, label_dir, rgb_dir, gradient_dir, is_random=False):
    """
    Sealing of loading training data.
    :param train_dir: halftoned images dir
    :param label_dir: gray images dir
    :return: two ndarray of training data and ground truth data
    """
    train_list = get_all_files(train_dir)
    label_list = get_all_files(label_dir)
    rgb_list = get_all_files(rgb_dir)
    gradient_list = get_all_files(gradient_dir)

    print('Training data dir:%s, including %d items.' % (train_dir, len(train_list)))
    print('Ground truth data dir:%s, including %d items.' % (label_dir, len(label_list)))
    print('Color data dir:%s, including %d items.' % (rgb_dir, len(rgb_list)))
    print('Gadient data dir:%s, including %d items.' % (gradient_dir, len(gradient_list)))

    assert len(train_list) == len(label_list), "Training data size doesn't equals ground truth data."

    if is_random is True:
        rnd_index = np.arange(len(train_list))
        np.random.shuffle(rnd_index)
        train_list = train_list[rnd_index]
        label_list = label_list[rnd_index]
        rgb_list = rgb_list[rnd_index]
        gradient_list = gradient_list[rnd_index]

    return [train_list, label_list, rgb_list, gradient_list]

def get_one(train_dir, label_dir, rgb_dir, gradient_dir, size):
    height,width = size
    # load contour images
    image_train = tf.read_file(train_dir)
    image_train = tf.image.decode_png(image_train, channels=1)
    image_train = tf.image.resize_images(image_train, [height, width])
    image_train = tf.cast(image_train, tf.float32) / 255.0

    # load RGB images
    gray_train = tf.read_file(label_dir)
    gray_train = tf.image.decode_png(gray_train, channels=3)
    gray_train = tf.image.resize_images(gray_train, [height, width])
    gray_train = tf.cast(gray_train, tf.float32) / 255.0

    # load  color
    rgb_train = tf.read_file(rgb_dir)
    rgb_train = tf.image.decode_png(rgb_train, channels=3)
    rgb_train = tf.image.resize_images(rgb_train, [height, width])
    rgb_train = tf.cast(rgb_train, tf.float32) / 255.0

    # load gradient
    gradient_train = tf.read_file(gradient_dir)
    gradient_train = tf.image.decode_png(gradient_train, channels=1)
    gradient_train = tf.image.resize_images(gradient_train, [height, width])
    gradient_train = tf.cast(gradient_train, tf.float32) / 255.0
    return image_train, gray_train, rgb_train, gradient_train

def get_train_batch(train_list, image_size, batch_size, capacity, is_random=False):
    """
    Sealing of getting batches for training.
    :param train_list:
    :param is_random:
    :param image_size: [height, width]
    :param batch_size:
    :param capacity: queue length
    :return: training batches
    """
    height, width = image_size
    filename_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # load contour images
    image_train = tf.read_file(filename_queue[0])
    image_train = tf.image.decode_png(image_train, channels=1)
    image_train = tf.image.resize_images(image_train, [height, width])
    image_train = tf.cast(image_train, tf.float32) / 255.0

    # load RGB images
    gray_train = tf.read_file(filename_queue[1])
    gray_train = tf.image.decode_png(gray_train, channels=3)
    gray_train = tf.image.resize_images(gray_train, [height, width])
    gray_train = tf.cast(gray_train, tf.float32) / 255.0

    # load  color
    rgb_train = tf.read_file(filename_queue[2])
    rgb_train = tf.image.decode_png(rgb_train, channels=3)
    rgb_train = tf.image.resize_images(rgb_train, [height, width])
    rgb_train = tf.cast(rgb_train, tf.float32) / 255.0

    # load gradient
    gradient_train = tf.read_file(filename_queue[3])
    gradient_train = tf.image.decode_png(gradient_train, channels=1)
    gradient_train = tf.image.resize_images(gradient_train, [height, width])
    gradient_train = tf.cast(gradient_train, tf.float32) / 255.0


    # get batches
    if is_random is True:
        image_train_batch, gray_train_batch, rgb_train_batch, gradient_train_batch = tf.train.shuffle_batch([image_train, gray_train, rgb_train, gradient_train],
                                                                     batch_size=batch_size,
                                                                     capacity=capacity,
                                                                     min_after_dequeue=500,
                                                                     num_threads=4)
    else:
        image_train_batch, gray_train_batch, rgb_train_batch, gradient_train_batch = tf.train.batch([image_train, gray_train, rgb_train, gradient_train],
                                                             batch_size=1,
                                                             capacity=capacity,
                                                             num_threads=1)

    return image_train_batch, gray_train_batch, rgb_train_batch, gradient_train_batch


def save_image(image, save_dir):
    """
    Save one image.
    :param image:  ndarray of image
    :param save_dir:
    :return:
    """
    img = Image.fromarray(image * 255)
    img.convert('RGB').save(save_dir)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     train_dir = 'E:\\Stanford_Dogs\\image_contour_train'
#     label_dir = 'E:\\Stanford_Dogs\\image_train'
#
#     train_list = get_train_list(train_dir, label_dir, is_random=True)
#     image_train_batch, label_train_batch =\
#         get_train_batch(train_list, [256, 256], 8, 1000, is_random=True)
#
#     sess = tf.Session()
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     try:
#         for step in range(10):
#             if coord.should_stop():
#                 break
#
#             image_batch, label_batch = sess.run([image_train_batch, label_train_batch])
#             plt.subplot(1, 2, 1), plt.imshow(image_batch[0, :, :, 0], 'gray'), plt.title('Contour image')
#             plt.subplot(1, 2, 2), plt.imshow(label_batch[0, :, :, :]), plt.title('Color image')
#             plt.show()
#
#     except tf.errors.OutOfRangeError:
#         print('Done.')
#     finally:
#         coord.request_stop()
#
#     coord.join(threads=threads)
#     sess.close()