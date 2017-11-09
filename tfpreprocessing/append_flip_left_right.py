import tensorflow as tf


def append_flip_left_right(X, y):
    flipped = tf.map_fn(lambda img: tf.image.flip_left_right(img), X)
    X = tf.concatenate([X, flipped])
    y = tf.concatenate([y, y])

    return X, y