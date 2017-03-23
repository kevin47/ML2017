import pandas as pd
#import numpy as np
from itertools import chain
import sys
import time
from math import log

data = pd.read_csv('train.csv', encoding='big5')
N = 5652
feats = 18
epo = 8000
ans = {
	'w': pd.Series(0, index=range(feats)),
	'b': 0,
	'rw': pd.Series(0, index=range(feats)),
	'rb': 0,
	'rate': 1
}
train_data = pd.DataFrame(index=range(feats+1), columns=range(N))
########################################
test_data = pd.read_csv('test_X_mod.csv', encoding='big5').stack()
########################################

def process_data():
	global data, test_data
	test_data = test_data.replace({'NR': 0}, regex=True)

	data.drop(data.columns[[1]], axis=1, inplace=True)
	data = data.replace({'NR': 0}, regex=True)
	data.columns = ['date', 'item'] + data.columns[2:].tolist()
	data = pd.melt(data, id_vars=['item', 'date'])
	data['date'] = pd.to_datetime(
			pd.to_datetime(data['date']).astype(int) +
			data['variable'].astype(int)*3600*1e09)
	data.drop('variable', axis=1, inplace=True)
	data = data.pivot(index='date', columns='item')
	data.columns = data.columns.droplevel(0)
	#data.columns = data.columns[:9].tolist() + data.columns[10:].tolist() + [data.columns[9]]
	data = data.astype(float)

	expand_data()

def expand_data():
	global data, train_data
	
	'''select only PM2.5'''
	tmp_data = data['PM2.5'].tolist()
	#tmp_dat2 = data['SO2'].tolist()
	for month in range(12):
		for begin_hour in range(471):
			index = month*480 + begin_hour
			#tmp = tmp_data[index : index+9]
			#train_data[month*471 + begin_hour] = tmp + [log(i) if i > 0 else -3 for i in tmp]
			#train_data[month*471 + begin_hour] = tmp + [tmp_data[index+9]]
			#print train_data[month*471 + begin_hour].tolist()
			#train_data[month*471 + begin_hour] = [(c+1000000) for c in tmp_dat2[index : index+9]] + tmp_data[index : index+10]
			#train_data[month*471 + begin_hour] = tmp_data[index : index+10]
			train_data[month*471 + begin_hour] = list(chain.from_iterable(
				(c, c**2) for c in tmp_data[index : index+9])) + [tmp_data[index+9]]
	train_data = train_data.transpose()
	
def loss_function(y, m, x):
	#assert type(y) == pd.core.series.Series
	tmp = y-m
	loss = (tmp**2).sum()/N
	db = tmp*2/N
	dw = x.mul(db, axis=0).sum()
	return loss, dw, db.sum()

def train(epoch, features, a):
	x = train_data.drop([features], axis=1)
	m = train_data[features]
	for i in range(1, epoch+1):
		y = x.dot(a['w']) + a['b']
		loss, dw, db = loss_function(y, m, x)
		#assert type(a['rw']) == pd.core.series.Series
		a['rw'] = a['rw'] + dw**2
		a['rb'] += db**2
		#assert type(a['w']) == pd.core.series.Series
		a['w'] = a['w'] - a['rate']*dw/(a['rw']**0.5)
		a['b'] -= a['rate']*db/(a['rb']**0.5)
		#if i%1000 == 999: print '{}th loss: {}'.format(i, loss**0.5) 

def test(a):
	global test_data, feats
	print 'id,value'
	for i in range(240):
		#curr = test_data[i*18 + 9][2:].astype(float).append(test_data[i*18 + 12][2:].astype(float))
		#curr = test_data[i*18 + 9][2:].astype(float)
		curr = pd.DataFrame(list(chain.from_iterable(
			(c, c**2) for c in test_data[i*18 + 9][2:].astype(float))))
		curr.index = range(feats)
		print 'id_{},{}'.format(i, (curr.transpose().dot(a['w'])+a['b'])[0])



process_data()
train(epo, feats, ans)
test(ans)
