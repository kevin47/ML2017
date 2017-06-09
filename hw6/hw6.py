#!/usr/bin/env python3
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot, add
from keras.layers import Input, Embedding, merge
from keras.layers.core import Reshape, Dense
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd
import sys

def read():
	'''
	data = np.array(pd.read_csv('train.csv'))
	usr = data[:, 1]
	mov= data[:, 2]
	y_train = data[:, 3]
	'''
	data = np.array(pd.read_csv(sys.argv[1]+'test.csv'))
	test_usr = data[:, 1]
	test_mov = data[:, 2]

	return test_usr, test_mov
	#return usr, mov, y_train, test_usr, test_mov

def main():
	#usr, mov, y_train, test_usr, test_mov = read()
	test_usr, test_mov = read()

	# normalize
	#mean = y_train.mean()
	#std = y_train.std()
	#y_train = (y_train-mean)/std
	'''
	input_dim = 6400
	output_dim = 100
	
	usr_id_input = Input(shape=(1,))
	x = Embedding(input_dim=input_dim, output_dim=output_dim)(usr_id_input)
	x = Reshape((output_dim,))(x)

	mov_id_input = Input(shape=(1,))
	y = Embedding(input_dim=3953, output_dim=output_dim)(mov_id_input)
	y = Reshape((output_dim,))(y)

	bu = Embedding(input_dim=input_dim, output_dim=1)(usr_id_input)
	bu = Reshape((1,))(bu)

	bm = Embedding(input_dim=input_dim, output_dim=1)(mov_id_input)
	bm = Reshape((1,))(bm)

	z = dot([x, y], axes=-1)
	z = add([z, bu, bm])

	model = Model(inputs=[usr_id_input, mov_id_input], outputs=[z])
	model.compile(optimizer='adam', loss='mse')
	

	model.fit([usr, mov], y_train, epochs=10, batch_size=1024)
	model.save('model.h5')
	'''
	model = load_model('model_saved.h5')
	pre = model.predict([test_usr, test_mov])

	f = open(sys.argv[2], 'w')
	f.write('TestDataID,Rating\n')
	for i in range(pre.shape[0]):
		#f.write(str(i+1) + ',' + str(pre[i][0]*std + mean) + '\n')
		f.write(str(i+1) + ',' + str(pre[i][0]) + '\n')
	f.close()
	#from IPython import embed
	#embed()

if __name__ == '__main__':
	main()
