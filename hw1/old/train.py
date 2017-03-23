import pandas as pd
import numpy as np
import sys
import time

#_items = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
data = pd.read_csv('train.csv', encoding='big5')
#train_data = pd.DataFrame(index=range(5652), columns=range(163))
train_data = pd.DataFrame(index=range(163), columns=range(5652))
test_data = pd.read_csv('test_X_mod.csv', encoding='big5').stack()
test_data = test_data.replace({'NR': 0}, regex=True)

def process_data():
	global data

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
	data.columns = data.columns[:9].tolist() + data.columns[10:].tolist() + [data.columns[9]]
	data = data.astype(float).as_matrix()

	#print data.head(15)

	start = time.time()
	expand_data()
	end = time.time()
	print 'expand_data() takes {} seconds'.format(end-start)


#@profile
def expand_data():
	global data, train_data
	'''
	data = data.as_matrix()
	for month in range(12):
		for begin_hour in range(471):
			for item in range(162):
				#total = (480*month + begin_hour)*18 + item

				#assert (month*471 + begin_hour) < 5652
				#assert (month*480 + begin_hour + item/18) < 5760
				#print month, begin_hour, item
				#sys.stdout.flush()
				#train_data[0][item] = float(0)
				#train_data[item][month*471 + begin_hour] = float(
				#	data.ix[month*480 + begin_hour + item/18][item%18])
				train_data[item][month*471 + begin_hour] = float(
					data[month*480 + begin_hour + item/18][item%18])
			#assert (month*471 + begin_hour) < 5652
			#assert (month*480 + begin_hour + 9) < 5760
			#print month, begin_hour
			train_data[162][month*471 + begin_hour] = float(data[month*480 + begin_hour + 9][17])
	'''
	for month in range(12):
		for begin_hour in range(471):
			index = month*480 + begin_hour
			train_data[month*471 + begin_hour] = np.append(
				data[index : index+9].flatten(), data[index+9][17])
			#if month == 0 and begin_hour == 0:
			#	print data[index : index+9].flatten(), [data[index+9][17]]
			#	print np.append(data[index : index+9].flatten(), data[index+9][17])
		



def loss_function(y, m, x):
	tmp = y-m
	loss = float((tmp**2).sum()/5652)
	db = float(tmp.sum()*2/5652)
	dw = (x*db).sum()/5652
	return loss, dw, db


rw, rb = 0, 0
def train(epoch):
	global w, b, rw, rb
	x = train_data.drop(['162'], axis=1)
	for i in range(1, epoch+1):
		ans = pd.DataFrame(x.values.dot(w.values)).add(b, fill_value=0)	
		loss, dw, db = loss_function(ans, train_data['162'].to_frame(name=0), x)
		#tmp= (dw/(i**0.5))
		dw.index = range(162)
		rw += (dw**2).sum()
		rb += db**2
		w = (w[0] - 1e-2*dw/((rw/i)**0.5)).to_frame(name=0)
		#w = (w[0] - dw/(i**0.5)/((rw/i)**0.5)).to_frame(name=0)
		#b -= db/(i**0.5)/((rb/i)**0.5)
		b -= 1e-2*db/((rb/i)**0.5)
		print w.head()
		print dw.head()
		print loss, db


def test():
	global test_data, w, b
	print 'id,value'
	for i in range(240):
		curr = []
		for j in range(18):
			curr += map(float, test_data[i*18 + j][2:])
		print 'id_{},{}'.format(i, (pd.DataFrame(curr).transpose().dot(w)+b[0][0])[0][0])





process_data()
'''
##############################
train_data = pd.read_csv('temp.csv')
train_data.drop(train_data.columns[[0]], axis=1, inplace=True)
##############################
w = pd.DataFrame(0, index=range(162), columns=[1])
w.columns = [0]
b = pd.DataFrame(0, index=range(5652), columns=[1])
b.columns = [0]

train(2000)
#x = train_data.drop(['162'], axis=1)
#ans = pd.DataFrame(x.values.dot(w.values)).add(b, fill_value=0)	
#loss, dw, db = loss_function(ans, train_data['162'].to_frame(name=0), x)
#tmp= (dw/(1**0.5))
#tmp.index = range(162)
#w = (w[0]-tmp).to_frame(name=0)
#b -= db/(1**0.5)

###############################
#w = pd.read_csv('w.csv')
#w.drop(w.columns[[0]], axis=1, inplace=True)
#b = pd.read_csv('b.csv')
#b.drop(b.columns[[0]], axis=1, inplace=True)
###############################
test()'''
#curr = []
#for j in range(18):
#	curr += map(float, test_data[0*18 + j][2:])
#print pd.DataFrame(curr).dot(w)+b
