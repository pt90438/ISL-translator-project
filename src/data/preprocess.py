import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
    
def load_images(directory):
    images = []
    labels = []
    outputs = 0
    uniq_labels = sorted(os.listdir(directory))

    for idx, label in enumerate(uniq_labels):
        print(label, " is ready to load")
        outputs = outputs + 1
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (128, 128))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)

    print(images.shape)
    images = images.astype('float32')/255.0
    labels = tf.keras.utils.to_categorical(labels)
    return(images, labels, outputs)

directory = "C:/Users/HP/Desktop/coding/fmlab/ISL translator project/dataset/Training"
images, labels, number_of_outputs = load_images(directory)   # file path containing training image data folder