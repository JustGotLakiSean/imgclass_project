import numpy as np
import os
import tensorflow as tf

from tensorflow import keras

batch_size = 32
img_height = 180
img_width = 180

print("80% of the images for training")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Complink\Desktop\imgclass_project\animals",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print("----------")

print("20% of the images for validation")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\Complink\Desktop\imgclass_project\animals",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print("----------")

class_names = train_ds.class_names
print(class_names)

new_model = tf.keras.models.load_model('save_model/image_classification_model')

#new_model.summary()

img = keras.preprocessing.image.load_img(
    r"C:\Users\Complink\Desktop\imgclass_project\dog_img.jpg",
    target_size=(180, 180)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = new_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
