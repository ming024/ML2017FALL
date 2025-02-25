from __future__ import print_function
import sys
import numpy as np
import csv
import os
import random
import csv
import tensorflow
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Reshape, GRU, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras import backend as K

num_words = 20000
train_labeled_path = "training_label.txt"
test_path = sys.argv[1]

print("===================Data preparation====================")
infile = open(train_labeled_path, 'r')
#train = [line.strip('\n') for line in infile]
train = [line.strip('\n') for line in infile]
labeled_sample = (len(train))
y_labeled = np.asarray([row[0] for row in train]).astype(int).reshape(-1, 1)
train = [row[10:] for row in train]
infile.close()

testfile = open(test_path, 'r')
next(testfile)
test = [line.strip('\n').split(',', 1)[1] for line in testfile]
testfile.close()

text = train + test

tokenizer = Tokenizer(num_words = num_words,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))#dictionary

l=0
for i in sequences:
	if len(i) > l:
		l = len(i)
#l = 12

x_labeled = pad_sequences(tokenizer.texts_to_sequences(train), maxlen=l)
x_test = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=l)
#x_semi = pad_sequences(tokenizer.texts_to_sequences(semi), maxlen=l)

#model = load_model('pretrained_model(26)')
#predict = model.predict(x_semi)
#y_labeled =y_labeled.tolist()
#for i  in range(0, len(semi)):
#	sentence = semi[i]
#	value = predict[i]
#	if value[0] > 0.9:
#		train += [sentence]
#		y_labeled += [1]
#	elif value[0] < 0.1:
#		train += [sentence]
#		y_labeled += [0]
#y_labeled = np.asarray(y_labeled).reshape(-1, 1)




text = train + test

for i in range(0, len(text)):
	iterator = 0
	while(iterator < len(text[i]) - 3):
		if text[i][iterator] == text[i][iterator + 1] and text[i][iterator + 1] == text[i][iterator + 2] and text[i][iterator + 2] == text[i][iterator + 3]:
			text[i] = text[i][:iterator] + text[i][iterator + 1:]
			iterator -= 1
		#elif iterator < len(text[i]) - 2 and text[i][iterator] == 'l' and text[i][iterator + 1] == 'y' and text[i][iterator + 2] == ' ':
		#	text[i] = text[i][:iterator] + text[i][iterator + 2:]
		#	iterator -= 1
		#elif iterator < len(text[i]) - 3 and text[i][iterator] == 'i' and text[i][iterator + 1] == 'n' and text[i][iterator + 2] == 'g' and text[i][iterator + 3] == ' ':
		#	text[i] = text[i][:iterator] + text[i][iterator + 3:]
		#	iterator -= 1
		#elif iterator < len(text[i]) - 3 and text[i][iterator] == text[i][iterator + 1] and text[i][iterator + 2] == 'y' and text[i][iterator + 3] == ' ':
		#	text[i] = text[i][:iterator + 1] + text[i][iterator + 3:]
		#	iterator -= 1
		#elif iterator < len(text[i]) - 1 and text[i][iterator] == 's' and text[i][iterator + 1] == ' ':
		#	text[i] = text[i][:iterator] + text[i][iterator + 1:]
		#	iterator -= 1
		elif iterator < len(text[i]) - 2 and text[i][iterator] == '\'' and text[i][iterator - 1] == ' ' and text[i][iterator + 1] == ' ':
			text[i] = text[i][:iterator - 1] + text[i][iterator + 2:]
			iterator -= 2
		iterator += 1
	#print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bline ' + str(i), end = '', flush = True)

l=0
for i in sequences:
	if len(i) > l:
		l = len(i)
#l = 12

x_labeled = pad_sequences(tokenizer.texts_to_sequences(train), maxlen=l)
x_test = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=l)

print("====================Glove Word2Vec=====================")
embedding_matrix = np.zeros((num_words + 1, 100))

dictfile = open('dict.csv', 'r')
dict_list = [line.strip('\n').split(',') for line in dictfile]
dictionary = dict((row[0], row[1]) for row in dict_list)
dictfile.close()

vecfile = open('vec.csv?dl=1', 'r')
word_vector = [line.strip('\n').split(',') for line in vecfile]
vecfile.close()

for word, i in tokenizer.word_index.items():
	idx = dictionary.get(word)
	if idx != None:
		embedding_vector = word_vector[int(idx)]
		if embedding_vector is not None and i < num_words + 1:
		# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
embedding_matrix = embedding_matrix.astype(float)

print("================Predicting and savining================")
model = load_model('trained_model(1)')
result = model.predict(x_test)
prefile = open("result(1).csv", 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['id', 'label'])
dataid = 0
for value in result:
	temp = int(round(result[dataid][0]))
	writer.writerow([str(dataid), str(temp)])
	dataid += 1
prefile.close()
probfile = open("prob(1).csv", 'w+')
writer = csv.writer(probfile, delimiter = ',', lineterminator = '\n')
dataid = 0
for value in result:
	temp = result[dataid][0]
	writer.writerow([str(temp)])
	dataid += 1
probfile.close()
