#! /usr/bin/bash

if [ ! -d 'grid_search/log' ]
then
    mkdir -p grid_search/log
fi

if [ ! -d 'grid_search/result' ]
then
    mkdir -p grid_search/result
fi


for alpha in `seq 0.01 0.02 0.21`
do
    for L1 in `seq 0.2 0.4 2.0`
    do
        for L2 in `seq 0.2 0.4 2.0`
        do
            echo $alpha, $L1, $L2
            sed  --in-place "s/alpha = .* #/alpha = $alpha #/" fast_solution_stdin.py
            sed  --in-place  "s/L1 = .* #/L1 = $L1 #/" fast_solution_stdin.py
            sed  --in-place "s/L2 = .* #/L2 = $L2 #/" fast_solution_stdin.py
            sed  --in-place "s/submission = .* #/submission = \\'grid_search\\/result\\/submission_${alpha}_${L1}_${L2}.csv\\' #/" fast_solution_stdin.py
            pypy fast_solution_stdin.py > grid_search/log/log_${alpha}_${L1}_${L2}.txt
        done
    done
done
