import numpy as np
N = 9;

def loss(prediction, target):
	return prediction - target;

def get_RMSE(array, coefficients):
	target = array[N:, 9];#

	return RMS(loss(prediction(array, coefficients), target));

def RMS(vector):
	sum = 0;
	count = 0;
	for element in vector:
		count += 1;
		sum += (element * element);
	return np.sqrt(sum / count);

def prediction(array, coefficients):
	temp_array = np.array([]);
	prediction = np.zeros(array.shape[0] - N);

	for count in range(0, array.shape[0] - N):
		temp_array = array[count:count + N].flatten();
		prediction[count] = \
			np.dot(temp_array, coefficients[0:17 * N]) + coefficients[17 * N];#
	return prediction;

def gradient_descent(array, coefficients, init_learning_rate, randnum, samplenum, derivatives_SS):
	target = np.transpose(array)[9, randnum + N:randnum + samplenum];#
	sample = array[randnum:randnum + samplenum];
	derivatives = np.zeros(17 * N + 1);#
	flattened_sample = np.array([[]]).reshape(samplenum - N, 0);
	bias = np.array([0]);

	for i in range (0, N):
		flattened_sample = \
			np.append(flattened_sample, sample[i:samplenum - N + i], axis = 1);
	
	result = prediction(sample, coefficients);
	lossnum = loss(result, target);

	derivatives= 2 * np.dot(np.transpose(flattened_sample), lossnum);
	bias[0] = 2 * np.sum(lossnum);
	derivatives = np.append(derivatives, bias, axis = 0);
	
	derivatives_SS += derivatives ** 2;
	learning_rate = init_learning_rate / np.sqrt(derivatives_SS);

	coefficients -= learning_rate * derivatives;
	
	return (coefficients, learning_rate, derivatives_SS);

def close_form(array):
	new_array = np.array([[]]).reshape(0, array.shape[1] * N);
	temp_array = np.array([[]]).reshape(1, 0);
	target = np.transpose(array)[9][N:array.shape[0]];#

	for i in range (0, array.shape[0]-N):
		for j in range(0, N):
			temp_array = np.append(temp_array, array[i + j:i + j + 1, 0:], axis = 1);
		
		new_array = np.append(new_array, temp_array, axis = 0);
		temp_array = np.array([[]]).reshape(1, 0);
		

	coefficients = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(new_array), new_array)), \
		np.transpose(new_array)), target);
	coefficients = np.append(coefficients, [[0]]);

	print '==========clossform=========='
	print coefficients;
	return coefficients;
