from __future__ import print_function
import sys
import numpy as np
import csv
import os
import random
import csv
import pickle
import tensorflow
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Reshape, GRU, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras import backend as K

num_words = 20000
#path = os.environ.get("GRAPE_DATASET_DIR")
#train_labeled_path = os.path.join(path, "data/training_label.txt")
#train_nolabeled_path = os.path.join(path, "data/training_nolabel.txt")
#test_path = os.path.join(path, "data/testing_data.txt")
train_labeled_path = sys.argv[1]
train_nolabeled_path = sys.argv[2]

print("===================Data preparation====================")
infile = open(train_labeled_path, 'r')
#train = [line.strip('\n') for line in infile]
train = [line.strip('\n').strip('\'').strip('\"') for line in infile]
labeled_sample = (len(train))
y_labeled = np.asarray([row[0] for row in train]).astype(int).reshape(-1, 1)
train = [row[10:] for row in train]
infile.close()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

l=37
#l = 12

x_labeled = pad_sequences(tokenizer.texts_to_sequences(train), maxlen=l)

a = b = c = d = e = 0
for i in tokenizer.word_counts.values():#appearance count
	if i < 6:
		a += 1
	if i < 5:
		b += 1
	if i < 4:
		c += 1
	if i < 3:
		d += 1
	if i < 2:
		e += 1
print ('appear < 6 : ' + str(a))
print ('appear < 5 : ' + str(b))
print ('appear < 4 : ' + str(c))
print ('appear < 3 : ' + str(d))
print ('appear < 2 : ' + str(e))

indices = np.arange(labeled_sample)
np.random.shuffle(indices)
x_labeled = x_labeled[indices]
y_labeled = y_labeled[indices]
x_labeled_valid = x_labeled[:int(labeled_sample / 10)]
y_labeled_valid = y_labeled[:int(labeled_sample / 10)]
x_labeled = x_labeled[int(labeled_sample / 10):]
y_labeled = y_labeled[int(labeled_sample / 10):]


print("====================Model Building=====================")
model = Sequential()
model.add(Embedding(input_dim = num_words, output_dim = 64, input_length=l))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be smaller than num_words (vocabulary size).
# now model.output_shape == (None, l, 64), where None is the batch dimension.

model.add(Conv1D(128, 3, padding = 'same', activation='relu'))
#model.add(Bidirectional(LSTM(units = 128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, unroll=True)))
#model.add(LSTM(units = 128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, unroll=True))
model.add(Bidirectional(LSTM(units = 64, dropout=0.2, recurrent_dropout=0.2, unroll=True)))
#model.add(Reshape((None, 1, 128)))
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(MaxPooling1D(5))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(MaxPooling1D(35))
#model.add(Flatten())
#model.add(Dense(64, activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

print("=======================Training========================")
early_stop = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, patience = 5)
print_acc = LambdaCallback(on_epoch_end = lambda epoch, \
	logs: print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' % (epoch, logs['val_acc'], epoch, logs['acc'])))
checkpoint = ModelCheckpoint(filepath = 'trained_model', monitor = 'val_acc', save_best_only = True)
model.fit(x_labeled, y_labeled, batch_size = 100, epochs = 30, callbacks = [print_acc, checkpoint, early_stop], validation_data=(x_labeled_valid, y_labeled_valid))
score, acc = model.evaluate(x_labeled_valid, y_labeled_valid, batch_size = 100)
print('Test score:', score)
print('Test accuracy:', acc)
