import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = Sequential()

base_model.add(Conv2D(64, (9, 9), input_shape=(100, 100, 3), padding='same'))
base_model.add(BatchNormalization())
base_model.add(Activation('relu'))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(Conv2D(128, (5, 5), padding='same'))
base_model.add(BatchNormalization())
base_model.add(Activation('relu'))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Conv2D(256, (3, 3), padding='same'))
base_model.add(BatchNormalization())
base_model.add(Activation('relu'))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Conv2D(512, (3, 3), padding='same'))
base_model.add(BatchNormalization())
base_model.add(Activation('relu'))

base_model.add(ZeroPadding2D((1, 1)))
base_model.add(Conv2D(512, (3, 3), padding='same'))
base_model.add(BatchNormalization())
base_model.add(Activation('relu'))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(Flatten())
base_model.add(Dense(64))
base_model.add(BatchNormalization())
base_model.add(Activation('relu'))
base_model.add(Dropout(0.5))

base_model.add(Dense(128))
base_model.add(BatchNormalization())
base_model.add(Activation('relu'))
base_model.add(Dropout(0.5))

base_model.add(Dense(1))
base_model.add(BatchNormalization())
base_model.add(Activation('sigmoid'))

base_model.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   shear_range=0.2,
                                   zoom_range=[0.8, 1.2],
                                   horizontal_flip=True,
                                   #                                     vertical_flip = True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'E:\Saaswath\optical glcoma\SOURCE CODE/dataset',
                                                 target_size=(100, 100),
                                                 batch_size=64,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(r'E:\Saaswath\optical glcoma\SOURCE CODE/dataset',
                                            target_size=(100, 100),
                                            batch_size=64,
                                            class_mode='binary')
my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=4, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('my_model2.h5',
                                       verbose=1, save_best_only=True, save_weights_only=False)
]


base_model.fit(training_set, epochs=20,
               validation_data=test_set, callbacks=my_callbacks)
base_model.save('my_model2.h5')
