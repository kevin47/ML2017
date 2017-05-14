#!/usr/bin/env python3
import word2vec
import nltk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.manifold import TSNE

def train():
	print('word2vec')
	#word2vec.word2phrase('books/all.txt', 'books/all-phrases.txt')
	word2vec.word2vec('books/all.txt', 'books/all.bin', size=50, iter_=550, window=25, negative=20)

def test():
	n = 300
	model = word2vec.load('books/all.bin')
	'''
	print('read')
	f = open('books/all.txt').read()
	print('tag')
	tokens = [x for x in nltk.word_tokenize(f) if len(x) > 1 and x.isalpha()]
	tagged = nltk.pos_tag(tokens)
	tagged = [x for x in tagged if x[1] in ['JJ', 'NNP', 'NN', 'NNS']] 
	'''

	print('tsne')
	tsne = TSNE()
	trans = tsne.fit_transform(model.vectors[:n])

	'''
	(x, y), w = [trans[i], w for i, w in enumerate(model.vocab) if ((nltk.pos_tag([w])[0][1] in ['JJ', 'NNP', 'NN', 'NNS']) and w.isalpha() and len(w) > 1)]
	plt.scatter(x, y)
	texts = [plt.text(xx, yy, ww) for xx, yy, ww in zip(x, y, w)]

	'''
	texts = []
	for i, word in enumerate(model.vocab[:n]):
		if nltk.pos_tag([word])[0][1] in ['JJ', 'NNP', 'NN', 'NNS'] and word.isalpha() and len(word) > 1:
			x, y = trans[i]
			texts.append(plt.text(x, y, word))
			plt.scatter(x, y)
	adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
	plt.savefig('word2vec.png')


def main():
	train()
	test()

if __name__ == '__main__':
	main()

