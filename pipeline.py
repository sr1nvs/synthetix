import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

H=256
W=256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    path = "dataset/"
    images = sorted(glob(os.path.join(path, "images/train/*"))) 
    masks = sorted(glob(os.path.join(path, "segments/train/*")))

    split_size = int(len(images) * split_size)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    # returns the path of all images matching the path/*

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256)) # resize
    x = x / 255.0 # normalize
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1) # add channel dimension
    x = x.astype(np.float32)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

path = "dataset/"
images, masks = load_data(path)
print(f"Images: {len(images)} - Masks: {len(masks)}")

dataset = tf_dataset(images, masks)

for x, y in dataset:
        x = x[0] * 255
        y = y[0] * 255

        x = x.numpy()
        y = y.numpy()

        cv2.imwrite("image.png", x)

        y = np.squeeze(y, axis=-1)
        cv2.imwrite("mask.png", y)
