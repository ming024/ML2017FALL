import sys
import numpy as np
from numpy.linalg import inv 
import csv
import functions

x_train = functions.readfile(sys.argv[1], 'r')
y_train = functions.readfile(sys.argv[2], 'r')
x_test = functions.readfile(sys.argv[3],'r')

for i in range (0, x_train.shape[1] - 1):
	ave = np.mean(x_train[:, i], axis = 0)
	dev = np.std(x_train[:, i], axis = 0)
	x_train[:, i] = (x_train[:, i] - ave) / dev
	x_test[:, i] = (x_test[:, i] - ave) / dev

for i in [0, 1, 3, 4, 5]:
	x_train = np.append(x_train, x_train[:, i].reshape(x_train.shape[0], 1) ** 2, axis = 1)
for i in [0, 1, 3, 4, 5]:
	x_test = np.append(x_test, x_test[:, i].reshape(x_test.shape[0], 1) ** 2, axis = 1)

x_train_0 = np.array([]).reshape(0, x_train.shape[1])
x_train_1 = np.array([]).reshape(0, x_train.shape[1])
y_train_0 = np.array([]).reshape(0, y_train.shape[1])
y_train_1 = np.array([]).reshape(0, y_train.shape[1])

for i in range(0, x_train.shape[0]):
	if y_train[i] == 0:
		x_train_0 = np.append(x_train_0, x_train[i:i + 1, :], axis = 0) 
		y_train_0 = np.append(y_train_0, y_train[i:i + 1, :], axis = 0) 
	else:
		x_train_1 = np.append(x_train_1, x_train[i:i + 1, :], axis = 0) 
		y_train_1 = np.append(y_train_1, y_train[i:i + 1, :], axis = 0) 

N0 = x_train_0.shape[0]
N1 = x_train_1.shape[0]
ave0 = np.mean(x_train_0, axis = 0).astype(np.float64)
ave1 = np.mean(x_train_1, axis = 0).astype(np.float64)
cov0 = np.zeros((111, 111))#np.cov(np.transpose(x_train_0))
cov1 = np.zeros((111, 111))#np.cov(np.transpose(x_train_1))
for v in x_train_0:
	cov0 += np.dot(np.transpose((v - ave0).reshape(1, 111)), (v - ave0).reshape(1, 111))
for v in x_train_1:
	cov1 += np.dot(np.transpose((v - ave1).reshape(1, 111)), (v - ave1).reshape(1, 111))
cov0 /= N0
cov1 /= N1
cov = (N0 * cov0 + N1 * cov1) / (N0 + N1)
w = np.transpose(np.dot(ave0 - ave1, inv(cov)))
b = (-0.5) * np.dot(np.dot(ave0, inv(cov)), np.transpose(ave0)) + \
	(0.5) * np.dot(np.dot(ave1, inv(cov)), np.transpose(ave1)) + np.log(N0 / N1)

y_train_predict = [value for value in (np.dot(x_train, w) + b)]
for i in range(0, len(y_train_predict)):
	if y_train_predict[i] > 0:
		y_train_predict[i] = 0
	else:
		y_train_predict[i] = 1
count = 0
for i in range(0, y_train.shape[0]):
	if y_train_predict[i] == y_train[i]:
		count += 1
print 'Training_accuracy : ' + str(float(count) / y_train.shape[0])

result = [value for value in (np.dot(x_test, w) + b)]
prefile = open(sys.argv[4], 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['id', 'label'])
dataid = 1
for value in result:
	writer.writerow([str(dataid), '1' if value < 0 else '0'])
	dataid += 1
prefile.close()
