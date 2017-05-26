#!/usr/bin/env python3
import numpy as np
import string
import sys
import pickle
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


test_path = sys.argv[1]
output_path = sys.argv[2]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 200
batch_size = 128


################
###   Util   ###
################
def read_data(path,training):
	print ('Reading data from ',path)
	with open(path,'r') as f:
	
		tags = []
		articles = []
		tags_list = []
		
		f.readline()
		for line in f:
			if training :
				start = line.find('\"')
				end = line.find('\"',start+1)
				tag = line[start+1:end].split(' ')
				article = line[end+2:]
				
				for t in tag :
					if t not in tags_list:
						tags_list.append(t)
			   
				tags.append(tag)
			else:
				start = line.find(',')
				article = line[start+1:]
			
			articles.append(article)
			
		if training :
			assert len(tags_list) == 38,(len(tags_list))
			assert len(tags) == len(articles)
	return (tags,articles,tags_list)

def f1_score(y_true,y_pred):
	thresh = 0.4
	y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
	tp = K.sum(y_true * y_pred,axis=-1)
	
	precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
	recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
	return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def main():

	### read training and testing data
	(_, X_test,_) = read_data(test_path,False)
	
	### tokenizer for all data
	tokenizer = pickle.load(open('tokenizer.p', 'rb'))
	word_index = tokenizer.word_index

	### convert word sequences to index sequence
	test_sequences = tokenizer.texts_to_sequences(X_test)

	### padding to equal length
	test_sequences = pad_sequences(test_sequences,maxlen=306)
	
	model = load_model('model1.h5', custom_objects={'f1_score': f1_score})

	Y_pred = model.predict(test_sequences)
	thresh = 0.4
	with open(output_path,'w') as output:
		print ('\"id\",\"tags\"',file=output)
		Y_pred_thresh = (Y_pred > thresh).astype('int')
		for index,labels in enumerate(Y_pred_thresh):
			labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
			labels_original = ' '.join(labels)
			print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
	main()
