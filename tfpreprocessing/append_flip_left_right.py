import tensorflow as tf


def append_flip_left_right(X, y):
    flipped = tf.map_fn(lambda img: tf.image.flip_left_right(img), X)
    X = tf.concat([X, flipped], axis=0)
    y = tf.concat([y, y], axis=0)

    return X, y