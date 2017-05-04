#!/usr/bin/env python
# -- coding: utf-8 --

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from keras.utils import np_utils
from IPython import embed

def load_data():
	data = pd.read_csv('train.csv')
	y_train = np_utils.to_categorical(np.array(data['label']), 7)

	x_train = data['feature']
	x_train = np.array(map(lambda x: map(float, x.split()), x_train))
	x_train = np.reshape(x_train, (x_train.shape[0], 48, 48, 1))/255
	x_train = x_train

	return x_train, y_train

def main():
	emotion_classifier = load_model("model_best.h5")

	x_train, y_train = load_data()
	private_pixels = x_train[7120].reshape((1,48,48,1))
	
	plt.figure()
	plt.imshow(private_pixels.reshape((48,48)), cmap='gray')
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig('original.png', dpi=100)

	input_img = emotion_classifier.input
	img_ids = ["image ids from which you want to make heatmaps"]

	val_proba = emotion_classifier.predict(private_pixels)
	pred = val_proba.argmax(axis=-1)
	target = K.mean(emotion_classifier.output[:, pred])
	grads = K.gradients(target, input_img)[0]
	fn = K.function([input_img, K.learning_phase()], [grads])

	heatmap = fn([private_pixels, False])[0]
	heatmap = (heatmap-heatmap.mean())/heatmap.std()
	heatmap = abs(np.clip(heatmap, -1, 1))
	heatmap = heatmap.reshape((48,48))

	thres = 0.5
	see = private_pixels.reshape(48, 48)
	#embed()
	see[np.where(heatmap <= thres)] = np.mean(see)


	plt.figure()
	plt.imshow(heatmap, cmap=plt.cm.jet)
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig('heat.png', dpi=100)

	plt.figure()
	plt.imshow(see,cmap='gray')
	plt.colorbar()
	plt.tight_layout()
	fig = plt.gcf()
	plt.draw()
	fig.savefig('saliency.png', dpi=100)

if __name__ == "__main__":
	main()
