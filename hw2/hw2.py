import pandas as pd
import numpy as np
import sys
import time
from math import log
import math
from IPython import embed


N = 32561
#N = 471*int(sys.argv[1])
epo = 1000
train_data = pd.read_csv(sys.argv[1])
train_data = pd.concat([train_data, train_data**2, train_data**3, train_data**0.5], axis=1)
feats = len(train_data.columns)
#adagrad
#ans = {
#	'w': pd.Series(0, index=range(feats+1)),
#	'rw': pd.Series(0, index=range(feats+1)),
#	'rate': 0.1
#}

#adam
ans = {
	'w': pd.Series(0, index=range(feats+1)),
	'm': pd.Series(0, index=range(feats+1)),
	'v': pd.Series(0, index=range(feats+1))
}

train_data.columns = range(feats)
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = train_data.sub(mean).div(std)
train_data[feats] = pd.Series(1, index=range(N))
#embed()

train_ans = pd.core.series.Series(pd.read_csv(sys.argv[2], header=None, usecols=[0])[0])

test_data = pd.read_csv(sys.argv[3])
test_data = pd.concat([test_data, test_data**2, test_data**3, test_data**0.5], axis=1)
test_data.columns = range(feats)
test_data = test_data.sub(mean).div(std)
test_data[feats] = pd.Series(1, index=range(N))

def sigmoid(x):
	if (x > 700): return 1
	if (x < -700): return 0
	return 1.0/(1+math.e**(-x))

def loss_function(y, m, xt):
	dw = xt.dot(y-m)
	tmp = pd.Series(y.round())
	n = m.dot(tmp) + (1-m).dot(1-tmp)
	loss = float(n)/len(y)
	return loss, dw

def train(epoch, features, a, x, xt, m):
	for i in range(1, epoch+1):
		y = x.dot(a['w'])
		loss, dw = loss_function(y.map(sigmoid), m, xt)
		#adagrad
		#a['rw'] = a['rw'] + dw**2
		#a['w'] = a['w'] - a['rate']*dw/(a['rw']**0.5)

		#adam
		a['m'] = 0.9*a['m'] + 0.1*dw
		a['v'] = 0.999*a['v'] + 0.001*(dw**2)
		a['w'] = a['w'] - 0.1*a['m']/(1-(0.9**i))/(((a['v']/(1-(0.999**i)))**0.5)+1e-8)
		#print 'accuracy:', loss

def test(a, features):
	global test_data
	ans = test_data.dot(a['w'])
	ans = ans.map(lambda x: int(round(sigmoid(x))))
	ans.index = range(1, len(ans)+1)
	ans.to_csv(sys.argv[4], index_label=['id'], header=['label'])

	return ans

train(epo, feats+1, ans, train_data, train_data.transpose(), train_ans)
test(ans, feats)
