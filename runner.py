import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import cv2
import pickle
import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import glob

columns = ['pixelsX', 'pixelsY', 'age', 'benign_malignant', 'diagnosis', 'melanocytic', 'sex', 'filename', 'localisation']
dataframe = pd.DataFrame(columns=columns)
dataframe_file_path = 'Data/meta_dataframe.df'

reload = input('Reload Dataframe? [y/n]')
if not reload == 'y':
    with open(dataframe_file_path, 'rb') as f:
        dataframe = pickle.load(f)
else:
    for filename in os.listdir('Data/descriptions'):
        print('Reading file: %s' % filename)
        with open('Data/descriptions/' + filename) as data_file:
            data = json.load(data_file)
            try:
                loc = data['meta']['unstructured']['localization']
            except KeyError:
                loc = 'unknown'
            try:
                diagnosis = data['meta']['clinical']['diagnosis']
            except KeyError:
                diagnosis = 'unknown'
            try:
                melanocytic = data['meta']['clinical']['melanocytic']
            except KeyError:
                melanocytic = 'unknown'

            row = [[data['meta']['acquisition']['pixelsX'], data['meta']['acquisition']['pixelsY'],
                   data['meta']['clinical']['age_approx'], data['meta']['clinical']['benign_malignant'],
                   diagnosis, melanocytic, data['meta']['clinical']['sex'], filename,
                   loc]]
            temp_df = pd.DataFrame(row, columns=columns)
            dataframe = dataframe.append(temp_df, ignore_index=True)
            b_m = data['meta']['clinical']['benign_malignant']
            image_src = 'Data/images/' + filename + '.jpg'
            if os.path.exists(image_src) and b_m:
                if 'benign' in b_m:
                    os.rename(image_src, 'Data/images/benign/' + filename + '.jpg')
                elif 'malignant' in b_m:
                    os.rename(image_src, 'Data/images/malignant/' + filename + '.jpg')
                elif 'indeterminate' in b_m:
                    os.rename(image_src, 'Data/images/indeterminate/' + filename + '.jpg')
    with open(dataframe_file_path, 'wb') as f:
        pickle.dump(dataframe, f)


print(set(dataframe['benign_malignant'].tolist()))
image_filenames = dataframe['filename'].tolist()



model = Sequential()
model.add(BatchNormalization(input_shape=(224, 224, 3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
        height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
        horizontal_flip=True)  # randomly flip images horizontally
val_datagen = ImageDataGenerator(rescale=1./255)

epochs = 10
batch_size = 32

train_generator = train_datagen.flow_from_directory(
        'Data/images/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        'Data/images/validate',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

checkpointer = ModelCheckpoint(filepath='Data/bestaugmented.model',
                               verbose=1, save_best_only=True)


def get_steps(dir, batch_size):
    list = os.listdir(dir)
    number_files = len(list)
    return number_files/batch_size

steps_per_epoch = get_steps('Data/images/train', batch_size)
validation_steps = get_steps('Data/images/validate', batch_size)

### Using Image Augmentation
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                    epochs=epochs,validation_data=val_generator, callbacks=[checkpointer],
                    validation_steps=validation_steps, verbose=1)

