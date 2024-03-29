import os
import csv

# Load in the training metadata
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Divide the data into training and validation sets. Use 20%
# of the data for validation.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from math import ceil
from random import shuffle
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization

# Use a generator to deliver batches of samples to Keras for training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2BGR)
                center_angle = float(batch_sample[3])

                # In order to augment the data and give the network balanced images to look
                # at, flip each image left/right, as well as the steering angle.
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle

                images.append(center_image)
                angles.append(center_angle)
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size, sized to use up available memory on laptop
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
# Normalize the data to be in the range (-0.5, 0.5)
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(160, 320, 3),
        output_shape=(160, 320, 3)))

# The network architecture below is straight from NVIDIAs DAVE2 paper
# for deep learning applied to self driving vehicles, referenced here:
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

# Convolution layers
model.add(Convolution2D(24,(5,5), subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,(5,5), subsample=(2,2),  activation='relu'))
model.add(Convolution2D(48,(5,5), subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))

# Fully connected layers
model.add(Flatten())

# This is the only modification I made to the network: adding 
# dropout before the first fully connected layer. This improved
# the network's tendency to overfit a bit for me.
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use the Adam optimizer

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Set up a keras callback to checkpoint the model every time the accuracy
# improved.
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Run the training. Note we multiplied the number of train/valid samples
# by two to take into account the augmentation we do in the generator.
model.fit_generator(train_generator,
            steps_per_epoch=ceil(2*len(train_samples)/batch_size), 
            validation_data=validation_generator,
            validation_steps=ceil(2*len(validation_samples)/batch_size),
            epochs=5, verbose=1, use_multiprocessing=True, callbacks=callbacks_list)

model.save('model.h5')
exit()
