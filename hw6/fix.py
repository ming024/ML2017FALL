import numpy as np
import sys
import csv
import os
import pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

tag = 'pcakmeans-400'
DEEPQ = False

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])
pre_path = str(sys.argv[3])
model_path = 'PTK/trained_model'
tsne_path = 'PTK/tsne'
agg_path = 'PTK/agg'
kmeans_path = 'kmeans20002_'
fig_path = 'PTK/scatter.png'
if DEEPQ:
	path = os.environ.get("GRAPE_DATASET_DIR")
	train_path = os.path.join(path, "image.npy")
	test_path = os.path.join(path, "test_case.csv")
	pre_path = 'hw6/result.csv'
	model_path = 'hw6/trained_model'
	tsne_path = 'hw6/tsne'
	agg_path = 'hw6/agg'
	kmeans_path = 'hw6/kmeans'


def compare(arr1, arr2):
	count = 0
	for i in range(100):
		if arr1[i] == arr2[i + 5000]:
			count += 1
	return True if count > 50 else False

kmeans = pickle.load(open(kmeans_path + '0', 'rb'))[:140000].tolist()
flag = [True]

for i in range(round(140000 / 140000) - 1):
	#tsne = pickle.load(open(tsne_path + str(i + 1), 'rb'))
	kmeans_raw = pickle.load(open(kmeans_path + str(i + 1), 'rb'))
	#kmeans_raw = kmeans_raw.predict(tsne)
	#tsne = pickle.load(open(tsne_path + str(i), 'rb'))
	kmeans_raw_prev = pickle.load(open(kmeans_path + str(i), 'rb'))
	#kmeans_raw_prev = kmeans_raw_prev.predict(tsne)

	if compare(kmeans_raw, kmeans_raw_prev):
		flag += [flag[i]]
		if flag[i]:
			kmeans += [value for value in kmeans_raw[:140000]]
		else:
			kmeans += [not value for value in kmeans_raw[:140000]]
	else:
		flag += [False if flag[i] else True]
		if flag[i]:
			kmeans += [not value for value in kmeans_raw[:140000]]
		else:
			kmeans += [value for value in kmeans_raw[:140000]]

testfile = open(test_path, 'r')
next(testfile)
test = [(int(line.strip('\n').split(',')[1]), int(line.strip('\n').split(',')[2])) for line in testfile]
for i in range(len(test)):
	test[i] = 1 if (kmeans[test[i][0]] == kmeans[test[i][1]]) else 0

kmeansfile = open(kmeans_path + '_' + tag, 'w+')
writer = csv.writer(kmeansfile, delimiter = ',', lineterminator = '\n')
for value in kmeans:
	writer.writerow([str(int(value))])
kmeansfile.close()

prefile = open(pre_path, 'w+')
writer = csv.writer(prefile, delimiter = ',', lineterminator = '\n')
writer.writerow(['ID', 'Ans'])
dataid = 0
count = 2
for value in test:
	if value == 1:
		count += 1
	writer.writerow([str(dataid), str(value)])
	dataid += 1
print(count)
prefile.close()
