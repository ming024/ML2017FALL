from __future__ import print_function
import sys
import numpy as np
import csv
import os
import random
import csv
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, LambdaCallback

print("====================Data preparation=====================")
infile = open(sys.argv[1], 'r')
next(infile)
train = np.asarray([line.strip('\n').split(',') for line in infile])
y_train_raw = train[:, 0].astype(int).reshape(-1, 1)
x_train = np.asarray([line.split(' ') for line in train[:, 1]]).astype(float).reshape(-1, 48, 48, 1)
y_train = np.zeros((y_train_raw.shape[0], 7))
for i in range (0, y_train.shape[0]):
	y_train[i][y_train_raw[i]] = 1
x_train_valid = x_train[::10]
y_train_valid = y_train[::10]
x_train = np.asarray([x_train[i + 1:10 + i] for i in range(0, x_train.shape[0] - 10, 10)]).reshape(-1, 48, 48, 1)
y_train = np.asarray([y_train[i + 1:10 + i] for i in range(0, y_train.shape[0] - 10, 10)]).reshape(-1, 7)
x_train_valid = np.vstack((np.vstack((x_train_valid, x_train_valid)), x_train_valid)) 
y_train_valid = np.vstack((np.vstack((y_train_valid, y_train_valid)), y_train_valid)) 
x_train = np.vstack((np.vstack((x_train, x_train)), x_train)) 
y_train = np.vstack((np.vstack((y_train, y_train)), y_train))

train_generator = image.ImageDataGenerator(
#	featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True, \
#	zca_epsilon=1e-6, \
	rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, \
	shear_range=0.1, zoom_range=0.1, channel_shift_range=0.1, fill_mode='nearest', cval=0., \
	horizontal_flip=True, vertical_flip=False,  rescale=None, preprocessing_function=None, \
	data_format='channels_last')
train_generator.fit(x_train)
train_valid_generator = image.ImageDataGenerator(
#	featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True, \
#	zca_epsilon=1e-6, \
	rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, \
	shear_range=0.1, zoom_range=0.1, channel_shift_range=0.1, fill_mode='nearest', cval=0., \
	horizontal_flip=True, vertical_flip=False,  rescale=None, preprocessing_function=None, \
	data_format='channels_last')
train_valid_generator.fit(x_train_valid)

infile.close()

print('===========================Keras model building==========================')
model = Sequential()
model.add(Convolution2D(64, kernel_size = (7, 7), padding = 'valid', input_shape = (48, 48, 1), activation = 'relu'))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Convolution2D(64, kernel_size = (5, 5), activation = 'relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))	
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))	
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))	

#model.add(Convolution2D(64, kernel_size = (3, 3), activation = 'relu'))
#model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))	
#model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))	
#model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))	

model.add(Convolution2D(128, kernel_size = (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))	
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))					
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Convolution2D(128, kernel_size = (3, 3), activation = 'relu'))	
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))	
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))	

model.add(Convolution2D(256, kernel_size = (3, 3), activation = 'relu'))				
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))	
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))	

model.add(Convolution2D(256, kernel_size = (3, 3), activation = 'relu'))				
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))	
model.add(Flatten())

model.add(Dense(output_dim = 1024, activation = 'relu'))
model.add(Dropout(0.5))
#model.add(Dense(output_dim = 256, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dropout(0.5))
#model.add(Dense(output_dim = 32, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(output_dim = 7, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adadelta', metrics = ['accuracy'])
model.summary()
print('===========================Keras model running==========================')

for i in range (0, epoch_num):
	early_stop = EarlyStopping(monitor='val_acc', min_delta=0.005, patience=15)
	training_result = model.fit_generator( \
		train_generator.flow(x_train, y_train, batch_size= 100), \
		steps_per_epoch=x_train.shape[0] / 100, epochs = 100, \
		callbacks = [early_stop], \
		validation_data = train_valid_generator.flow(x_train_valid, y_train_valid, batch_size= 100), validation_steps = x_train_valid.shape[0] / 100)

model.save('model')
