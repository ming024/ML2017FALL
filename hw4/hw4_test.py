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

test_path = sys.argv[1]
save_path = sys.argv[2]

num_words = 20000
l = 37

testfile = open(test_path, 'r')
next(testfile)
test = [line.strip('\n').split(',', 1)[1] for line in testfile]
testfile.close()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

x_test = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=l)

print("================Predicting and savining================")
model = load_model('trained_model')
result = model.predict(x_test)
prefile = open(save_path, 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['id', 'label'])
dataid = 0
for value in result:
	temp = int(round(result[dataid][0]))
	writer.writerow([str(dataid), str(temp)])
	dataid += 1
prefile.close()

probfile = open("prob.csv", 'w+')
writer = csv.writer(probfile, delimiter = ',', lineterminator = '\n')
dataid = 0
for value in result:
	temp = result[dataid][0]
	writer.writerow([str(temp)])
	dataid += 1
probfile.close()
