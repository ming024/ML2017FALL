import csv
import sys
import numpy as np

print('reading file : ', end = '', flush = True)
i = 1
print(str(i), end = '', flush = True)
infile = open('prob(' + str(i) + ').csv', 'r')
prob = np.asarray([line.strip('\n') for line in infile]).astype(float)
infile.close()
print('\b' * len(str(i)), end = '', flush = True)

#for i in list(set(np.arange(int(sys.argv[1])).tolist()) - set([0, 8, 9, 13])):
for i in [2]:
	print(str(i), end = '', flush = True)
	infile = open('prob(' + str(i) + ').csv', 'r')
	temp = np.asarray([line.strip('\n') for line in infile]).astype(float)
	prob += temp
	infile.close()
	print('\b' * len(str(i)), end = '', flush = True)
print('\b' * 15 + 'writing file...   \b\b\b', end = '', flush = True)

probfile = open(sys.argv[2], 'w+')
writer = csv.writer(probfile, delimiter = ',', lineterminator = '\n')
writer.writerow(['id', 'label'])
dataid = 0
for dataid in range(0, len(prob)):
	writer.writerow([str(dataid), str(int(round(prob[dataid] / 2)))])
	dataid += 1
probfile.close()
print()
