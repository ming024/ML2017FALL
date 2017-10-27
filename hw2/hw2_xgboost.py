import sys
import numpy as np
import csv
import xgboost
import sklearn
import functions

fold = 3
N = 10
x_train = functions.readfile(sys.argv[1], 'r')
y_train = functions.readfile(sys.argv[2], 'r')
x_test = functions.readfile(sys.argv[3],'r')

for i in [0, 1, 3, 4, 5]:
	x_train = np.append(x_train, x_train[:, i].reshape(x_train.shape[0], 1) ** 2, axis = 1)
for i in [0, 1, 3, 4, 5]:
	x_test = np.append(x_test, x_test[:, i].reshape(x_test.shape[0], 1) ** 2, axis = 1)

model1 = xgboost.XGBClassifier(colsample_bytree = 0.05, gamma = 1,learning_rate = 0.35, max_delta_step = 0,max_depth = 4, \
	min_child_weight = 1.5, n_estimtors = 692.008824,subsample = 0.7)
model2 = xgboost.XGBClassifier(colsample_bytree = 0.45, gamma = 1.5,learning_rate = 0.2, max_delta_step = 0,max_depth = 5, \
	min_child_weight = 2.5, n_estimtors = 792.008824,subsample = 0.75)
model3 = xgboost.XGBClassifier(colsample_bytree = 0.75, gamma = 2.8,learning_rate = 0.15, max_delta_step = 0,max_depth = 6, \
	min_child_weight = 1., n_estimtors = 192.008824,subsample = 0.9)
model4 = xgboost.XGBClassifier(colsample_bytree = 0.15, gamma = 1.7,learning_rate = 0.50, max_delta_step = 0,max_depth = 7, \
	min_child_weight = 0.9, n_estimtors = 892.008824,subsample = 0.85)
model5 = xgboost.XGBClassifier(colsample_bytree = 0.03, gamma = 0.4,learning_rate = 0.45, max_delta_step = 0,max_depth = 8, \
	min_child_weight = 7.5, n_estimtors = 292.008824,subsample = 0.8)
model6 = xgboost.XGBClassifier(colsample_bytree = 0.25, gamma = 0.1,learning_rate = 0.55, max_delta_step = 0,max_depth = 9, \
	min_child_weight = 10, n_estimtors = 992.008824,subsample = 0.7)
model7 = xgboost.XGBClassifier(colsample_bytree = 0.15, gamma = 2.2,learning_rate = 0.25, max_delta_step = 0,max_depth = 6, \
	min_child_weight = 2, n_estimtors = 992.008824,subsample = 0.8)
model8 = xgboost.XGBClassifier(colsample_bytree = 0.35, gamma = 0.9,learning_rate = 0.35, max_delta_step = 0,max_depth = 6, \
	min_child_weight = 0.3, n_estimtors = 392.008824,subsample = 0.7)
model9 = xgboost.XGBClassifier(colsample_bytree = 0.65, gamma = 1.4,learning_rate = 0.15, max_delta_step = 0,max_depth = 5, \
	min_child_weight = 0.1, n_estimtors = 592.008824,subsample = 0.7)
model10 = xgboost.XGBClassifier(colsample_bytree = 0.25, gamma = 0.8,learning_rate = 0.4, max_delta_step = 0,max_depth = 3, \
	min_child_weight = 6, n_estimtors = 92.008824,subsample = 0.9)



def xgboostcv(max_depth, learning_rate, n_estimators, gamma, min_child_weight, max_delta_step, subsample, \
              colsample_bytree, silent=True, nthread=-1):
	model = xgboost.XGBClassifier(max_depth=int(max_depth), learning_rate=learning_rate, \
		n_estimators=int(n_estimators), silent=silent, nthread=nthread, gamma=gamma, \
		min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=subsample, \
		colsample_bytree=colsample_bytree)
	model.fit(x_train, y_train)
	y_predict = np.asarray([round(value) for value in model.predict(x_train)]).reshape(x_train.shape[0])
	return sklearn.metrics.accuracy_score(y_train, y_predict)

training_accuracy = np.zeros((fold, N))
testing_accuracy = np.zeros((fold, N))
model_weight = np.zeros((fold, N))
result =  np.zeros((x_test.shape[0], fold, N))
for i in range (0, fold):
	x_train_temp = np.append(x_train[:i * (x_train.shape[0] / fold), :],\
		 x_train[(i + 1) * (x_train.shape[0] / fold):, :], axis = 0)
	y_train_temp = np.append(y_train[:i * (x_train.shape[0] / fold), :],\
		 y_train[(i + 1) * (x_train.shape[0] / fold):, :], axis = 0)
	x_test_temp = x_train[i * (x_train.shape[0] / fold):(i + 1) * (x_train.shape[0] / fold), :]
	y_test_temp = y_train[i * (x_train.shape[0] / fold):(i + 1) * (x_train.shape[0] / fold), :]
	
	y_train_temp_predict = np.zeros((x_train_temp.shape[0], N))
	y_test_temp_predict = np.zeros((x_test_temp.shape[0], N))
	for j in range (0, N):
		if j == 0:
			model = model4#1
		elif j == 1:
			model = model6#2
		elif j == 2:
			model = model8#3
		elif j == 3:
			model = model4
		elif j == 4:
			model = model5
		elif j == 5:
			model = model6
		elif j == 6:
			model = model7
		elif j == 7:
			model = model8
		elif j == 8:
			model = model9
		else:
			model = model10

		model.fit(x_train_temp, y_train_temp)
		y_train_temp_predict[:, j] = model.predict(x_train_temp)
		y_test_temp_predict[:, j] = model.predict(x_test_temp)
		result[:, i, j] = model.predict(x_test)
		training_accuracy[i][j] = sklearn.metrics.accuracy_score(y_train_temp, y_train_temp_predict[:, j])
		testing_accuracy[i][j] = sklearn.metrics.accuracy_score(y_test_temp, y_test_temp_predict[:, j])
		model_weight[i][j] = np.exp(3 / (1 - testing_accuracy[i][j]))
		print model_weight[i][j]

	print 'Training accuracy ' + str(i + 1) + ' = ' + str(training_accuracy[i])
	print 'Testing accuracy ' + str(i + 1) + ' = ' + str(testing_accuracy[i])

	y_train_temp_predict = [round(value) for value in np.dot(y_train_temp_predict, model_weight[i, :] / np.sum(model_weight[i]))]
	y_test_temp_predict = [round(value) for value in np.dot(y_test_temp_predict, model_weight[i, :] / np.sum(model_weight[i]))]
	training_accuracy[i][0] = sklearn.metrics.accuracy_score(y_train_temp, y_train_temp_predict)
	testing_accuracy[i][0] = sklearn.metrics.accuracy_score(y_test_temp, y_test_temp_predict)
	print 'Training accuracy ' + str(i) + ' = ' + str(training_accuracy[i][0])
	print 'Testing accuracy ' + str(i) + ' = ' + str(testing_accuracy[i][0])
	print 'Current avarage ' + str(i) + ' = ' + str(np.mean(testing_accuracy[0:i+1, 0]))
	
print '\nAverage training_accuracy = ' + str(np.mean(training_accuracy[:, 0])) 	
print 'Average testing_accuracy = ' + str(np.mean(testing_accuracy[:, 0])) 
result = np.mean(result, axis = 1)
result = [round(value) for value in np.dot(result, np.mean(model_weight, axis = 0) / np.sum(model_weight) * fold)]
print 'fold = ' + '10' + '\tpower = ' + '3'

prefile = open(sys.argv[4], 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['id', 'label'])
dataid = 1
for value in result:
	writer.writerow([str(dataid), '1' if value > 0.5 else '0'])
	dataid += 1
prefile.close()
