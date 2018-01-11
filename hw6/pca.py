import sys
import os
import numpy as np
from skimage.io import imread, imread_collection, imsave
from skimage.transform import resize

train_path = str(sys.argv[1]) + '/*.jpg'
test_path = os.path.join(str(sys.argv[1]) + '/', str(sys.argv[2]))

collection = imread_collection(train_path)
train = np.zeros((len(collection.concatenate()), 600, 600, 3))
for i in range(len(train)):
	train[i] = collection[i]
del collection
train = train.reshape(train.shape[0], -1)
ave = np.mean(train, axis = 0)
train -= ave
test = imread(test_path).reshape(1, -1).astype(np.float64)
test -= ave
imsave('ave.jpg', ave.reshape(600, 600, 3).astype(np.uint8))

U, S, V = np.linalg.svd(np.transpose(train), full_matrices = False)
eigenvalues = S.copy()
eigenfaces = np.transpose(U).copy()

for i in range(4):
	temp = eigenfaces[i].copy()
	temp -= np.min(temp)
	temp /= np.max(temp)
	temp = (temp * 255)
	imsave('eigen(' + str(i) + ').jpg', temp.reshape(600, 600, 3).astype(np.uint8))
	print('Ratio(' + str(i) + ') = ' + str(eigenvalues[i] / np.sum(eigenvalues)))
'''
for i in range(0, 12, 3):
	weights = np.dot(train[i], U[:, :4])
	recon = np.dot(weights, eigenfaces[:4])
	recon = resize(recon.reshape(200, 200, 3), (600, 600, 3))
	recon = recon + ave

	recon -= np.min(recon)
	recon /= np.max(recon)
	recon = (recon * 255).astype(np.uint8)
	imsave('reconstruction(' + str(i) + ').jpg', recon)
'''
weights = np.dot(test, U[:, :4])
recon = np.dot(weights, eigenfaces[:4])
recon = recon + ave
recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon * 255).reshape(600, 600, 3).astype(np.uint8)
imsave('reconstruction.jpg', recon)

weights = np.dot(test, U[:, :])
recon = np.dot(weights, eigenfaces[:])
recon = recon + ave
recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon * 255).reshape(600, 600, 3).astype(np.uint8)
imsave('FULL_reconstruction.jpg', recon)
