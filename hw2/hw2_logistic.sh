#!/bin/bash

# $1 raw data
# $2 test data
# $3 provided train feature
# $4 provided train label
# $5 provided test feature
# $6 prediction

echo "training"
python2 hw2.py $3 $4 $5 $6
echo "done"
