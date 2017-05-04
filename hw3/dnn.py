#!/usr/bin/env python2
import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from PIL import Image

FF = open('dnn_history.csv', 'w')
class History(Callback):
	def on_train_begin(self,logs={}):
		self.tr_los=[]
		self.vl_los=[]
		self.tr_acc=[]
		self.vl_acc=[]

	def on_epoch_end(self,epoch,logs={}):
		self.tr_los.append(logs.get('loss'))
		self.vl_los.append(logs.get('val_loss'))
		self.tr_acc.append(logs.get('acc'))
		self.vl_acc.append(logs.get('val_acc'))
		FF.write('{},{},{},{}\n'.format(logs.get('loss'), logs.get('acc'), logs.get('val_loss'), logs.get('val_acc')))
		FF.flush()

x_test = 0

def expand(data):
	tmp = np.flip(data, 2)
	tmp = np.concatenate((data, tmp), axis=0)
	tmp2 = [np.reshape(Image.fromarray(d[:,:,0]).rotate(10), (48, 48, 1)) for d in tmp]
	tmp3 = [np.reshape(Image.fromarray(d[:,:,0]).rotate(350), (48, 48, 1)) for d in tmp]
	return np.concatenate((tmp, tmp2, tmp3), axis=0)

def load_data():
	data = pd.read_csv('train.csv')
	y_train = np_utils.to_categorical(np.array(data['label']), 7)

	x_train = data['feature']
	x_train = np.array(map(lambda x: map(float, x.split()), x_train))
	x_train = np.reshape(x_train, (x_train.shape[0], 48, 48, 1))/255
	x_train = x_train


	data = pd.read_csv('test.csv')
	x_test = data['feature']
	x_test = np.array(map(lambda x: map(float, x.split()), x_test))
	x_test = np.reshape(x_test, (x_test.shape[0], 48, 48, 1))/255
	x_test = x_test

	return x_train, y_train, x_test

def load_and_save():
	x_train, y_train, x_test = load_data()
	fx = open('x_train', 'w')
	fy = open('y_train', 'w')
	fxx = open('x_test', 'w')
	pickle.dump(x_train, fx)
	pickle.dump(y_train, fy)
	pickle.dump(x_test, fxx)

def read_data():
	fx = open('x_train', 'r')
	fy = open('y_train', 'r')
	fxx = open('x_test', 'r')
	x_train = pickle.load(fx)
	x_train = expand(x_train)
	
	y_train = pickle.load(fy)
	y_train = np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train), axis=0)
	
	x_test = pickle.load(fxx)

	return x_train, y_train, x_test

def train(model, x_train, y_train, batch_size):
	callback = ModelCheckpoint('dnn.h5', monitor='val_accuracy')
	history = History()
	model.fit(x_train, y_train, batch_size=batch_size, epochs=150, validation_split=0.2, callbacks=[callback, history])
	'''
	for i in range(11, 50):
		model.fit(x_train, y_train, batch_size=batch_size, epochs=50, validation_split=0.1, callbacks=[callback, history])
		model.save('model'+str((i+1)*50)+'.h5')
	'''

def test(x_test, batch_size):
	model = load_model('dnn.h5')
	pre = model.predict(x_test, batch_size)
	f = open('submit.csv', 'w')
	f.write('id,label\n')
	for i, a in enumerate(pre):
		ans = np.where(a == a.max())[0][0]
		f.write('{},{}\n'.format(i, ans))
	f.close()

def main():
	#x_train, y_train, x_test = load_data()
	#x_train = expand(x_train)
	#y_train = np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train), axis=0)
	x_train, y_train, x_test = read_data()
	rate = 0.1
	model = Sequential()
	base = 32
		
	model.add(Flatten(input_shape=(48, 48, 1)))
	model.add(Dense(units=400, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=400, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=400, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=400, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=400, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=400, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=7, activation='softmax'))
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
	model.save('dnn.h5')
	#model = load_model('model.h5')
	batch_size = 128
	train(model, x_train, y_train, batch_size)
	del model
	test(x_test, batch_size)

if __name__ == '__main__':
	#load_and_save()
	main()
