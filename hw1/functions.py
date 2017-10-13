import numpy as np
N = 9;

def loss(prediction, target):
	return prediction - target;

def get_RMSE(array, coefficients):
	target = array[N:, 2];#

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
			np.dot(temp_array, coefficients[0:4 * 2 * N]) + coefficients[4 * 2 * N];#
	return prediction;

def gradient_descent(array, coefficients, init_learning_rate, randnum, samplenum, derivatives_SS):
	target = np.transpose(array)[2, randnum + N:randnum + samplenum];#
	sample = array[randnum:randnum + samplenum];
	derivatives = np.zeros(4 * 2 * N + 1);#
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
