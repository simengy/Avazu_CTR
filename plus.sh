#! /bin/bash

modelName="model_2"
resultName="plus_submission_2.csv"

pypy fast_solution_plus.py train --train train.csv -o model/$modelName --alpha 0.011 --beta 2.0 --L1 1.2 --L2 0.8 --dropout 0.3 --bits 30 --n_epochs 5 --sparse --onlydays 141021,141022,141023,141024,141025,141026,141027,141028,141029,141030

pypy fast_solution_plus.py predict --test test.csv -i model/$modelName -p plus_result/$resultName 
