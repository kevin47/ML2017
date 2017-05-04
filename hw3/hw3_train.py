#!/usr/bin/env python2
import numpy as np
import pandas as pd
import sys
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

FF = open('history.csv', 'w')
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

x_test = 0

def expand(data):
	tmp = np.flip(data, 2)
	tmp = np.concatenate((data, tmp), axis=0)
	tmp2 = [np.reshape(Image.fromarray(d[:,:,0]).rotate(10), (48, 48, 1)) for d in tmp]
	tmp3 = [np.reshape(Image.fromarray(d[:,:,0]).rotate(350), (48, 48, 1)) for d in tmp]
	return np.concatenate((tmp, tmp2, tmp3), axis=0)

def load_data():
	data = pd.read_csv(sys.argv[1])
	y_train = np_utils.to_categorical(np.array(data['label']), 7)

	x_train = data['feature']
	x_train = np.array(map(lambda x: map(float, x.split()), x_train))
	x_train = np.reshape(x_train, (x_train.shape[0], 48, 48, 1))/255
	x_train = x_train

	return x_train, y_train

def train(model, x_train, y_train, batch_size):
	callback = ModelCheckpoint('model.h5', monitor='val_accuracy')
	history = History()
	model.fit(x_train, y_train, batch_size=batch_size, epochs=750, validation_split=0.2, callbacks=[callback, history])

def main():
	x_train, y_train = load_data()
	x_train = expand(x_train)
	y_train = np.concatenate((y_train, y_train, y_train, y_train, y_train, y_train), axis=0)
	rate = 0.3
	model = Sequential()
	base = 32
	model.add(Conv2D(base, (3, 3), input_shape=(48, 48, 1), activation='relu'))
	model.add(BatchNormalization())
	#model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(rate))
	model.add(Conv2D(base*2, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(rate))
	model.add(Conv2D(base*3, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(rate))
	model.add(Conv2D(base*4, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(rate))
	model.add(Conv2D(base*5, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(rate))
	
	model.add(Flatten())
	model.add(Dense(units=800, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=7, activation='relu'))
	model.add(Dropout(rate))
	model.add(Dense(units=7, activation='softmax'))
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
	model.save('model.h5')
	train(model, x_train, y_train, 128)

if __name__ == '__main__':
	main()
