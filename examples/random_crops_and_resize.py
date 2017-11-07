import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tfpreprocessing import random_crops_and_resize

examples_dir = os.path.dirname(__file__)

img1 = cv2.imread(os.path.join(examples_dir, 'data/cat.jpg'))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (100, 100))
img1 = img1.astype(np.float32)

img2 = cv2.imread(os.path.join(examples_dir, 'data/dog.jpg'))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (100, 100))
img2 = img2.astype(np.float32)

imgs = np.asarray([img1, img2])
print('Imgs shape: ' + str(imgs.shape))

with tf.Session() as sess:
    placeholder = tf.placeholder(tf.float32, [None, 100, 100, 3])
    tensor = random_crops_and_resize(placeholder, min_size=(75, 75), name='random_crops_and_resize')

    variants = []
    for i in range(3):
        cropped = sess.run(tensor, feed_dict={placeholder: imgs})
        variants.append(cropped)

fig, ax = plt.subplots(2, 4)
ax[0][0].imshow(imgs[0].astype(np.uint8))
ax[0][1].imshow(variants[0][0].astype(np.uint8))
ax[1][0].imshow(variants[1][0].astype(np.uint8))
ax[1][1].imshow(variants[2][0].astype(np.uint8))
ax[0][2].imshow(imgs[1].astype(np.uint8))
ax[0][3].imshow(variants[0][1].astype(np.uint8))
ax[1][2].imshow(variants[1][1].astype(np.uint8))
ax[1][3].imshow(variants[2][1].astype(np.uint8))
plt.show()