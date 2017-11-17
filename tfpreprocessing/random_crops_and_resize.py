import tensorflow as tf


def random_crop_and_resize(tensor, *, min_size):
    height, width, channels = [int(x) for x in tensor.shape]
    min_height, min_width = min_size

    crop_height = tf.random_uniform([], minval=min_height, maxval=height, dtype=tf.int32)
    crop_width = tf.random_uniform([], minval=min_width, maxval=width, dtype=tf.int32)

    cropped = tf.random_crop(tensor, [crop_height, crop_width, channels])
    resized = tf.image.resize_images(cropped, [height, width])
    tensorflow_pls = tf.reshape(resized, [-1, height, width, channels])
    print('SHAPE: ' + str(tensorflow_pls.shape))

    return tensorflow_pls


def random_crops_and_resize(tensor, *, min_size, name):
    return tf.map_fn(lambda img: random_crop_and_resize(img, min_size=min_size), tensor)