from __future__ import print_function
import sys
import numpy as np
import csv
import os
import random
import csv
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Dot, Input, Add, Concatenate
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras import backend as K

#normalize dot_normalize
#path = os.environ.get("GRAPE_DATASET_DIR")
#train_path = os.path.join(path, "data/train.csv")
#test_path = os.path.join(path, "data/test.csv")
#model_path = 'hw5/trained_model'
#pre_path = 'hw5/result.csv'
test_path = sys.argv[1]
model_path = 'trained_model5'
pre_path = sys.argv[2]

def rmse(y_true, y_pred): return K.sqrt(K.mean((y_pred - y_true) ** 2))

testfile = open(test_path, 'r')
next(testfile)
test = [line.strip('\n').split(',')[1:] for line in testfile]

test_usr = []
test_mov = []
for line in test:
	test_usr += [int(line[0])]
	test_mov += [int(line[1])]
test_usr = np.asarray(test_usr)
test_mov = np.asarray(test_mov)
testfile.close()

print("================Predicting and savining================")
model = load_model(model_path, custom_objects = {'rmse': rmse})
result = model.predict([test_usr, test_mov])
prefile = open(pre_path, 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['TestDataID', 'Rating'])
dataid = 1
for value in result:
	writer.writerow([str(dataid), str(value[0])])
	dataid += 1
prefile.close()
