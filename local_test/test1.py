import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime
import os
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

import tensorflow_cloud as tfc
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import pathlib

################################################
######Image classification based on d-00-W9-rs224-v00 dataset#####################


skip_training = False
batch_size = 15
img_height = 224
img_width = 224
IMG_SIZE = (224, 224)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/home/paul/GDES/Data/gdes/ds-00-w9-rs224-v00/train',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/home/paul/GDES/Data/gdes/ds-00-w9-rs224-v00/validation',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = 2

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Base model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.resnet.preprocess_input

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


prediction_layer = tf.keras.layers.Dense(1)


inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


base_model.trainable = False

model.summary()

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

if skip_training==False:

    epochs=100
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )
    base_model.trainable = True

    #Fine-tune from this layer onwards
    fine_tune_at = 100

    #Freeze all the layers before the fine tune layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    fine_tune_epochs = 30
    total_epochs = epochs + fine_tune_epochs
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=val_ds)

    model.save('model-test03')
else:
    model = tf.keras.models.load_model('model-test01')

print("Validation")
results = model.evaluate(val_ds)
print("val loss, vall acc:", results)


if skip_training == False:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']
    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']



    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0, 1.0])
    plt.plot([epochs - 1, epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 5.0])
    plt.plot([epochs - 1, epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()



#Retrieve a batch of images from the test set
image_batch, label_batch = val_ds.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)


plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
plt.show()


