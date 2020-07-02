import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

_URL = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/21401/1288832/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1592835297&Signature=LwRXKCMSqm%2BeV2onmT49MTTR2Nu3QvKQcByXDGj8c%2BBmdw1H4tQ7ZGWEqzYwPug%2FzTLKWkMbBLqo%2F%2BtXebiUBfNz8oDFyi86TJvvPrylzj8xlQe7ogE87lYD%2FqriU%2BSCUs6UsyJBSbve9JK7Gj79QYnF00LI6muIklhtAUlaTJID1OIMlpeKTGRxqoi9efPYwnzbLK2zsIdlK2ArYo%2FZLmYtH1AhadmpERhrXttYGR%2Fndo3j%2B6jnM3Qg8gl1FPF%2B6ZVBU5ckWdAmUADNNtzCbhwnbwRXVtQ8L5eubSf8PaPTbGhzlcxqwPhbn3%2BScY%2BosuXA5lk2CUm%2Fl%2FO1vT9Xxw%3D%3D&response-content-disposition=attachment%3B+filename%3Daiat-hackathon-1-2020.zip"

PATH = './Data'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')

train_covid_dir = os.path.join(train_dir, 'covid')
train_normal_dir = os.path.join(train_dir, 'normal')
train_pneumonia_dir = os.path.join(train_dir, 'pneumonia')
validation_covid_dir = os.path.join(validation_dir, 'covid')
validation_normal_dir = os.path.join(validation_dir, 'normal')
validation_pneumonia_dir = os.path.join(validation_dir, 'pneumonia')

num_covid_tr = len(os.listdir(train_covid_dir))
num_normal_tr = len(os.listdir(train_normal_dir))
num_pneumonia_tr = len(os.listdir(train_pneumonia_dir))

num_covid_val = len(os.listdir(validation_covid_dir))
num_normal_val = len(os.listdir(validation_normal_dir))
num_pneumonia_val = len(os.listdir(validation_pneumonia_dir))

total_train = num_covid_tr + num_normal_tr + num_pneumonia_tr
total_val = num_covid_val + num_normal_val + num_pneumonia_val

print('total training covid images:', num_covid_tr)
print('total training normal images:', num_normal_tr)
print('total training pneumonia images:', num_pneumonia_tr)

print('total testing covid images:', num_covid_val)
print('total testing normal images:', num_normal_val)
print('total testing pneumonia images:', num_pneumonia_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)