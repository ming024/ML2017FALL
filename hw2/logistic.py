import sys
import numpy as np
import csv
import functions

init_learning_rate = 0.01
batch_size = 100
epoch_num = 1500
regularization = 0.01

x_train = functions.readfile(sys.argv[1], 'r')
y_train = functions.readfile(sys.argv[2], 'r')
x_test = functions.readfile(sys.argv[3],'r')
weight = np.zeros(x_train.shape[1] + 5).reshape(x_train.shape[1] + 5, 1)
bias = np.zeros(batch_size).reshape(batch_size, 1)

for i in range (0, x_train.shape[1] - 1):
	ave = np.mean(x_train[:, i], axis = 0)
	dev = np.std(x_train[:, i], axis = 0)
	x_train[:, i] = (x_train[:, i] - ave) / dev
	x_test[:, i] = (x_test[:, i] - ave) / dev

for i in [0, 1, 3, 4, 5]:
	x_train = np.append(x_train, x_train[:, i].reshape(x_train.shape[0], 1) ** 2, axis = 1)
for i in [0, 1, 3, 4, 5]:
	x_test = np.append(x_test, x_test[:, i].reshape(x_test.shape[0], 1) ** 2, axis = 1)

derivatives = np.zeros(x_train.shape[1] + 1).reshape(x_train.shape[1] + 1, 1)
derivatives_SS = np.zeros(x_train.shape[1] + 1).reshape(x_train.shape[1] + 1, 1)
learning_rate = np.zeros(x_train.shape[1] + 1).reshape(x_train.shape[1] + 1, 1)

for count in range(0, epoch_num):
	random_num_table = np.arange(x_train.shape[0])
	np.random.shuffle(random_num_table)
	for batch_count in range (0, int(random_num_table.size / batch_size)):
		batch = np.asarray([x_train[i] for i in \
			random_num_table[batch_count * batch_size:(batch_count + 1) * batch_size]])
		target = np.asarray([y_train[i] for i in \
			random_num_table[batch_count * batch_size:(batch_count + 1) * batch_size]])
			
		prob = 1 / (1 + np.exp( -1* (np.dot(batch, weight) + bias)))
		derivatives[0:derivatives.shape[0] - 1] = \
			np.dot(np.transpose(batch), (prob - target)) + 2 * weight * regularization
		derivatives[derivatives.shape[0] - 1][0] = np.sum(prob - target)
		derivatives /= batch_size
		derivatives_SS += derivatives ** 2 + 10 ** (-20)
		learning_rate = init_learning_rate / np.sqrt(derivatives_SS) 

		weight[:, 0] -= learning_rate[0:learning_rate.shape[0] - 1, 0] * \
			derivatives[0:derivatives.shape[0] - 1, 0]
		bias -= learning_rate[learning_rate.shape[0] - 1][0] * \
			derivatives[derivatives.shape[0] - 1][0]


		if count% 20 == 0 and batch_count == 0:	
			loss = np.sum(np.absolute(1 / \
				(1 + np.exp( -1* (np.dot(x_train, weight) + bias[0]))) - y_train))
			print 'round' + str(count) +'\t' + str(loss)

prefile = open(sys.argv[4], 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['id', 'label'])
dataid = 1
for value in (np.dot(x_test, weight))[:, 0]:
	writer.writerow([str(dataid), '1' if value + bias[0] > 0 else '0'])
	dataid += 1

prefile.close()
