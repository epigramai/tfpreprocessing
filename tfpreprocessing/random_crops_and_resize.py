import tensorflow as tf

def random_crops_and_resize(tensor, *, min_size, name):
    height, width, channels = [int(x) for x in tensor.shape[1:]]
    min_height, min_width = min_size

    crop_height = tf.random_uniform([], minval=min_height, maxval=height, dtype=tf.int64)
    crop_width = tf.random_uniform([], minval=min_width, maxval=width, dtype=tf.int64)

    cropped = tf.map_fn(lambda x: tf.random_crop(x, [crop_height, crop_width, channels]), tensor, name=name + '/cropped')
    resized = tf.image.resize_images(cropped, [height, width])

    return resized