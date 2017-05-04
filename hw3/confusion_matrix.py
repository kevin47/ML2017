#!/usr/bin/env python
# -- coding: utf-8 --
from IPython import embed

from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from keras.utils import np_utils

def load_data():
	data = pd.read_csv('train.csv')
	y_train = np_utils.to_categorical(np.array(data['label']), 7)

	x_train = data['feature']
	x_train = np.array(map(lambda x: map(float, x.split()), x_train))
	x_train = np.reshape(x_train, (x_train.shape[0], 48, 48, 1))/255
	x_train = x_train

	return x_train, y_train


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
	"""
	This function prints and plots the confusion matrix.
	"""
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def main():
	emotion_classifier = load_model('model100.h5')
	np.set_printoptions(precision=2)
	x_train, y_train = load_data()
	sz = x_train.shape[0]/10
	predictions = emotion_classifier.predict(x_train[4*sz:5*sz], 128)
	pre = np.array(map(lambda x: np.where(x == x.max())[0][0], predictions))
	ans = np.array(map(lambda x: np.where(x == x.max())[0][0], y_train[4*sz:5*sz]))
	pre2 = [x for i, x in enumerate(pre) if x != ans[i]]
	ans2 = [x for i, x in enumerate(ans) if x != pre[i]]
	#embed()
	conf_mat = confusion_matrix(ans2, pre2)

	plt.figure()
	plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
	#plt.show()
	plt.savefig('confusion_matrix2.png')

main()
