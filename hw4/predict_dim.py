#!/usr/bin/env python3
import numpy as np
import pickle
import math
from sklearn.decomposition import PCA
import sys
from time import time

def test(n_err, n_std, err, std):
	return (np.abs(n_err-err).sum(axis=1) + np.abs(n_std-std).sum(axis=1)).argmin()+1

def main():
	t = time()
	with open('err0.74', 'rb') as f:
		err = np.array(pickle.load(f))
	with open('std0.74', 'rb') as f:
		std = np.array(pickle.load(f))
	data = np.load(sys.argv[1])
	print('load file takes {}'.format(time()-t))

	f = open(sys.argv[2], 'w')
	f.write('SetId,LogDim\n')

	N = 10000
	for index in data:
		t = time()
		n_err, n_std = [], []
		for i in range(data[index].shape[0]//N):
			nn_err, nn_std = [], []
			tmp_data = data[index][i*N : (i+1)*N]
			for j in range(13):
				pca = PCA(j*5) # 0 5 10 15 ... 55 60
				t_data = pca.fit_transform(tmp_data)
				i_data = pca.inverse_transform(t_data)

				sub = tmp_data-i_data
				nn_err.append(np.abs(sub).mean())
				nn_std.append(sub.std())
			n_err.append(nn_err)
			n_std.append(nn_std)
		n_err = np.array(n_err).mean(axis=0)
		n_std = np.array(n_std).mean(axis=0)
		print('{}th shape {} PCA takes {}'.format(index, data[index].shape, time()-t))
		
		val = math.log(test(n_err, n_std, err, std))
		f.write('{},{}\n'.format(index, val))
		print(index, val, '\n')
			
if __name__ == '__main__':
	main()
