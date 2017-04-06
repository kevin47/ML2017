#!/bin/bash

get_seeded_random(){
	openssl enc -aes-256-ctr -pass pass:$1 -nosalt </dev/zero 2>/dev/null
}

# $1 raw data
# $2 test data
# $3 provided train feature
# $4 provided train label
# $5 provided test feature
# $6 prediction

#acc=0
header=`head -1 $3`
for i in {1..3};
do
	echo "round $i:"
	tail -n +2 $3 | shuf --random-source=<(get_seeded_random $i) > shuffled_X
	shuf $4 --random-source=<(get_seeded_random $i) > shuffled_Y

	n=`wc -l $3 | awk '{print $1}'`
	t=$(($n/3))
	h=$(($n-$t))

	echo $header > head_X
	head -$h shuffled_X >> head_X
	head -$h shuffled_Y > head_Y
	echo $header > tail_X
	tail -$t shuffled_X >> tail_X
	tail -$t shuffled_Y > tail_Y

	echo "training"
	python2 hw2.py head_X head_Y tail_X $6
	
	echo "testing"
	tail -n +2 $6 > ans
	read s <<< $(paste -d ',' ans tail_Y | awk -F "," -v t=$t 'BEGIN{s = 0}{s += $2 == $3}END{print s/t}')
	echo "accuracy:" $s
	#acc=$(($acc+$s))

	rm -f shuffled_X shuffled_Y head_X head_Y tail_X tail_Y ans
	echo "done"
done

#echo "mean accuracy:" $acc

