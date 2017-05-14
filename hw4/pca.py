#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def openall():
	A = np.zeros((100, 4096))
	AV = np.zeros((64, 64))
	t = 0
	for i in 'ABCDEFGHIJ':
		for j in range(10):
			a = mpimg.imread('face/'+i+str(j).rjust(2, '0')+'.bmp').reshape(1, 4096)
			AV += a.reshape(64, 64)	
			A[t] = a
			t += 1
	AV = AV/t
	return A, AV

plt.axis('off')
cmap = plt.get_cmap('gray')
def plot(img, name):
	plt.imshow(img, cmap=cmap)
	plt.savefig(name)

def arrange(imgs, n):
	tmp = np.zeros((64*n, 64*n))
	for i in range(n):
		for j in range(n):
			tmp[i*64 : (i+1)*64, j*64 : (j+1)*64] = imgs[i*n+j].reshape(64, 64)
	return tmp
	
def reconstruct(img, v, dim):
	P = img.dot(v[:dim].T)
	R = P.dot(v[:dim])
	return R

def min_k(A, v, miu):
	loss = 1e20
	for k in range(1, 100):
		R = reconstruct(A, v, k)
		t_loss= (((A-R)**2).sum()/409600)**0.5/256
		#print(k, t_loss)
		if t_loss < loss:
			k_mn = k
			R_mn = R
			loss = t_loss
		if loss < 0.01:
			return k_mn, R_mn
	return k_mn, R_mn

def main():

	A, AV = openall()

	#average
	plot(AV, 'averageface.png')

	#eigenfaces
	miu = A.mean(axis=0, keepdims=True)
	A = A - miu
	svd = np.linalg.svd(A)
	v = svd[2]
	E = arrange(v, 3)
	plot(E, 'eigenfaces.png')
	
	#original
	O = arrange(A+miu, 10)
	plot(O, 'original.png')

	#project & recover
	R = reconstruct(A, v, 5)
	RE = arrange(R+miu, 10)
	plot(RE, 'recovered.png')
	
	#smallest k
	k, BR = min_k(A, v, miu)
	print('best k:', k)
	BRE = arrange(BR+miu, 10)
	plot(BRE, 'smallest_recover.png')


if __name__ == '__main__':
	main()
