#! /usr/bin/bash

log_dir='grid_search/log'
if [ ! -d $log_dir ]
then
    echo 'The directory does not exist!'
    exit 1
fi

grep 'Epoch 1 finished, validation logloss:' $log_dir/* |\
awk -F: '{print $1, $3}' |\
awk -F, '{print $1}'  |\
awk 'BEGIN{min = 100} {if(min > $2) {min = $2; file = $0}} END{print file}'
