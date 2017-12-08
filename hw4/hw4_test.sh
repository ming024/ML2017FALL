#!/bin/bash

wget https://www.dropbox.com/s/kie1byju7glrbvf/vec.csv?dl=1
python3 hw4_test.py $1 $2
python3 hw4_ensemble.py $1 $2
