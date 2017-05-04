#!/usr/bin/env python2
import sys
import numpy as np
import pandas as pd
from keras.models import load_model

def load_data():
	data = pd.read_csv(sys.argv[1])
	x_test = data['feature']
	x_test = np.array(map(lambda x: map(float, x.split()), x_test))
	x_test = np.reshape(x_test, (x_test.shape[0], 48, 48, 1))/255
	x_test = x_test

	return x_test

def test(x_test):
	model = load_model('model_best.h5')
	pre = model.predict(x_test)
	f = open(sys.argv[2], 'w')
	f.write('id,label\n')
	for i, a in enumerate(pre):
		ans = np.where(a == a.max())[0][0]
		f.write('{},{}\n'.format(i, ans))
	f.close()

def main():
	x_test = load_data()
	test(x_test)

if __name__ == '__main__':
	main()
