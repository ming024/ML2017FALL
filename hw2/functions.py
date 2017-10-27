import numpy as np

def readfile(filename, flag):
	infile = open(filename, flag)
	next(infile)
	return np.asarray([line.strip('\n').split(',') for line in infile]).astype(float)
