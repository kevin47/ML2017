import pandas as pd
import numpy as np
import sys
import time
from math import log
import math
from IPython import embed


train_data = pd.read_csv(sys.argv[1])
feats = len(train_data.columns)
train_data.columns = range(feats)

train_ans = pd.core.series.Series(pd.read_csv(sys.argv[2], header=None, usecols=[0])[0])
train_data[feats] = train_ans

class0 = train_data[train_data[feats] == 0]
class0 = class0.drop(feats, axis=1)
mean0 = class0.mean(axis=0)
cov0 = np.cov(class0.T)
noise = np.random.randn(len(cov0), len(cov0))*1e-5
cov0 += noise

class1 = train_data[train_data[feats] == 1]
class1 = class1.drop(feats, axis=1)
mean1 = class1.mean(axis=0)
cov1 = np.cov(class1.T)
cov1 += noise

test_data = pd.read_csv(sys.argv[3])
test_data.columns = range(feats)

def f(cov, data):
	D = len(cov)
	a = 1/((math.pi*2)**D) * 1/(((cov**2).sum())**0.5)
	b = -data.dot(np.linalg.inv(cov)).multiply(data).sum(axis=1)/2
	return a*(np.e**b)
	

def test(features):
	global test_data, cov0, cov1, mean0, mean1
	ans0 = f(cov0, test_data-mean0)
	ans1 = f(cov1, test_data-mean1)
	ans = (ans0 < ans1).astype(int)
	ans.index = range(1, len(ans)+1)
	ans.to_csv(sys.argv[4], index_label=['id'], header=['label'])
	return ans

test(feats)
