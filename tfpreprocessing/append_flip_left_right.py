import tensorflow as tf


def append_flip_left_right(X, y):
    flipped = tf.map_fn(lambda img: tf.image.flip_left_right(img), X)
    X = tf.concat([X, flipped])
    y = tf.concat([y, y])

    return X, y