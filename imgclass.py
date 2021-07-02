import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
## From this ##
#dataset_url = r"C:\Users\Complink\Desktop\imgclass_project\animals"
##data_dir = tf.keras.utils.get_file("animals", origin=r"C:\Users\Complink\Desktop\imgclass_project\animals" ,untar=True)
##data_dir = pathlib.Path(data_dir)
## TO HERE ##
## is used to "download" dataset ##

##image_count = len(list(data_dir.glob('*/*.jpg')))
##print(image_count)

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

# ----------<TEST>---------- #
# TESTING THE MODEL PREDICTION AND CONFIDENCE WITH NEW IMAGE #
##new_model = tf.keras.models.load_model('save_model/image_classification_model')
##
###new_model.summary()
##
##img = keras.preprocessing.image.load_img(
##    r"C:\Users\Complink\Desktop\imgclass_project\predict_panda.jpg",
##    target_size=(180, 180)
##)
##
##img_array = keras.preprocessing.image.img_to_array(img)
##img_array = tf.expand_dims(img_array, 0)
##
##predictions = new_model.predict(img_array)
##score = tf.nn.softmax(predictions[0])
##
##print(
##    "This image most likely belongs to {} with a {:.2f} percent confidence."
##    .format(class_names[np.argmax(score)], 100 * np.max(score))
##)
# ----------</TEST>---------- #

##
print("----------")
##
####plt.figure(figsize=(10, 10))
####for images, labels in train_ds.take(1):
####    for i in range(9):
####        ax = plt.subplot(3, 3, i + 1)
####        plt.imshow(images[i].numpy().astype("uint8"))
####        plt.title(class_names[labels[i]])
####        plt.axis("off")
####
####plt.show()
##
####for image_batch, labels_batch in train_ds:
####    print(image_batch.shape)
####    print(labels_batch.shape)
####    break
##
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = 3
##
#### Overfitting generally occurs when there are a small number of training
#### examples.
#### Data Augmentation is used to generate additional training data from the
#### existing examples by augmenting them using random transformations that
#### yield believable-looking images.
##
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)
##
#### Data Augmentation Visualization
####    plt.figure(figsize=(10, 10))
####    for images, _ in train_ds.take(1):
####        for i in range(9):
####            augmented_images = data_augmentation(images)
####            ax = plt.subplot(3, 3, i + 1)
####            plt.imshow(augmented_images[0].numpy().astype("uint8"))
####            plt.axis("off")
####
####    plt.show()
##
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("---------- Model Summary ----------")
model.summary()

print("Model Training")
epochs=15

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

##    print("VISUALIZED TRAINING RESULT")
##
##    acc = history.history['accuracy']
##    val_acc = history.history['val_accuracy']
##
##    loss = history.history['loss']
##    val_loss = history.history['val_loss']
##
##    epochs_range = range(epochs)
##
##    plt.figure(figsize=(8, 8))
##    plt.subplot(1, 2, 1)
##    plt.plot(epochs_range, acc, label='Training Accuracy')
##    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
##    plt.legend(loc='lower right')
##    plt.title('Training and Validation Accuracy')
##
##    plt.subplot(1, 2, 2)
##    plt.plot(epochs_range, loss, label='Training Loss')
##    plt.plot(epochs_range, val_loss, label='Validation Loss')
##    plt.legend(loc='upper right')
##    plt.title('Training and Validation Loss')
##    plt.show()

model.save('save_model/image_classification_model')
