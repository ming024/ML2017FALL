import sys
import numpy as np
import csv
import os
from keras.models import load_model

testfile = open(sys.argv[1], 'r')
next(testfile)
test = np.asarray([line.strip('\n').split(',') for line in testfile])
x_test = np.asarray([line.split(' ') for line in test[:, 1]]).astype(float).reshape(-1, 48, 48, 1)
testfile.close()

model = load_model('model' + str(11) + '?dl=1')
result = model.predict(x_test)
for i in range (12, 23):
	model = load_model('model' + str(i) + '?dl=1')
	result += model.predict(x_test)

prefile = open(sys.argv[2], 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['id', 'label'])
dataid = 0
for value in result:
	temp = result[dataid].tolist()
	writer.writerow([str(dataid), str(temp.index(max(temp)))])
	dataid += 1
prefile.close()
