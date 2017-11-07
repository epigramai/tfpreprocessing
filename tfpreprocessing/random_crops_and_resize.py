import tensorflow as tf

def random_crops_and_resize(tensor, *, min_size, name):
    print('Tensor shape: ' + str(tensor.shape))
    height, width, channels = [int(x) for x in tensor.shape[1:]]
    min_height, min_width = min_size

    cropped = tf.map_fn(lambda x: tf.random_crop(x, [min_height, min_width, channels]), tensor, name=name + '/cropped')
    resized = tf.image.resize_images(cropped, [height, width])

    return resized