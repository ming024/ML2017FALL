#!/bin/bash

for i in {1..1}
do
	echo "Execution num : $i"
	python logistic.py $3 $4 $5 $6 $i
done
