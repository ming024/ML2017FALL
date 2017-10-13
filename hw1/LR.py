import sys
import numpy as np
import scipy
import pandas
import csv
import random
from scipy import stats

import functionsL

N = 9;
samplenum = 200;
init_learning_rate = np.array([]);
train_num = 0;
init_rate = [0.01];

infile = open('train.csv', 'r');
temp_array = np.array([[]]).reshape(0, 24);
array = np.array([[]]).reshape(17, 0);#
row_count = -1;

for row in csv.reader(infile):
	if (row_count != -1 and row_count % 18 != 10):
		temp_array = np.append(temp_array, np.delete(np.asarray(row), [0, 1, 2]).reshape(1, 24), axis = 0).astype(np.float);
	
	if row_count > 16:
		row_count = -1;
		array = np.append(array, temp_array, axis = 1);
		temp_array = np.array([[]]).reshape(0, 24);	

	row_count += 1;

for i in range (0, array.shape[1]):
	if array[9][i] == -1:
		for j in range (0, i):
			if(array[9][j - i] != -1):
				prev = array[9][j - i];
				break;
		
		for j in range (i, array.shape[1]):
			if(array[9][j] != -1):
				next = array[9][j];
				break;

		array[9][i] = (prev + next) / 2;

array = np.transpose(array);

coefficients = np.array([]);
previousfile = open('previous_coefficients_L', 'r');
for row in previousfile:
	coefficients = np.append(coefficients, np.asarray(float(row)));
if len(coefficients) == 0:
	coefficients = np.zeros(17 * N + 1);

derivatives_SS = np.zeros(17 * N + 1);#
for i in range (0, 17 * N + 1):#
	init_learning_rate = np.append(init_learning_rate, init_rate, axis = 0);
learning_rate = init_learning_rate;

for i in range (0, train_num):
	sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b')
	sys.stdout.write('round ');  sys.stdout.write(str(i));
	
	randnum = int(random.random() * (240 - (samplenum + 1))) + int(random.random() * 12) * 240;

	(coefficients, learning_rate, derivatives_SS) = \
		functionsL.gradient_descent(array, coefficients, init_learning_rate, \
		randnum, samplenum, derivatives_SS);

	if(i % 200 == 0):
		print '\tRMSE\t' + str(functionsL.get_RMSE(array,coefficients));
	
sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b')
print functionsL.get_RMSE(array,coefficients);

testfile = open(sys.argv[1], 'r');
temp_array = np.array([[]]).reshape(0, N);
array = np.array([[]]).reshape(0, 17 * N);#
row_count = 0;

for row in csv.reader(testfile):
	if row_count % 18 != 10:
		temp_array = np.append(temp_array, \
			np.delete(np.asarray(row), np.arange(11-N)).reshape(1, N), \
			axis = 0).astype(np.float);
	
	if row_count > 16:
		row_count = -1;

		for i in range (0, temp_array.shape[1]):
			if temp_array[9][i] == -1:
				if i == 0:
					prev = next = array[9][i + 1];
				elif i == temp_array.shape[1] - 1:
					prev = next = array[9][i - 1];
				else:
					prev = array[9][i - 1];
					next = array[9][i + 1];

				array[9][i] = (prev + next) / 2;

		array = np.append(array, np.transpose(temp_array).reshape(1, 17 * N), axis = 0);#
		temp_array = np.array([[]]).reshape(0, N);	

	row_count += 1;

PM25_prediction = np.dot(array, coefficients[0:17 * N]);#
for data in PM25_prediction:
	data += coefficients[17 * N];#

dataid = 0;
ofile = open(sys.argv[2], 'w+');
writer = csv.writer(ofile, delimiter = ',', lineterminator = '\n');
writer.writerow(['id', 'value']);
for value in PM25_prediction:
	writer.writerow(['id_'+str(dataid), str(value)]);
	dataid += 1;

previousfile = open('previous_coefficients_L', 'w+');
for value in coefficients:
	previousfile.write(str(value) + '\n');

testfile.close();
ofile.close();
previousfile.close();
