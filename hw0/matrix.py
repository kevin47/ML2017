#!/usr/bin/env python2
import numpy as np
import sys

fa = open(sys.argv[1], 'r')
fb = open(sys.argv[2], 'r')

def parse_matrix(A):
	for i, lines in enumerate(A):
		lines = lines[:-1].split(',')
		A[i] = lines
	return A

A = np.array(parse_matrix(fa.readlines())).astype(int)
B = np.array(parse_matrix(fb.readlines())).astype(int)

ans = np.sort(A.dot(B).flatten()).tolist()

f = open('ans_one.txt', 'w')
f.write('\n'.join(map(str, ans)))
f.write('\n')
