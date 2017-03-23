#!/bin/bash

echo id,item,0,1,2,3,4,5,6,7,8 > test_X_mod.csv
cat $2 >> test_X_mod.csv
python2 hw1.py 12 $1 > $3

rm test_X_mod.csv
