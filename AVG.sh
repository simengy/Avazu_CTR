#! /bin/bash

if [ $# -eq 2 ]
then
        N=$1
        prefix=$2
else
    N=0
    prefix="plus_log/log_CV.txt"
fi

echo
echo "Last epochs = $N"
echo "File name is $prefix"
echo
cat ${prefix}* | grep -B 1 "Epoch $N" | grep 'holdout logloss:'  | awk -F, '{print $2}' | awk -F: 'BEGIN{sum = 0.0} {sum = sum + $2} END{print NR, sum/NR}' 
