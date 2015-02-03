#! /bin/bash

ls run_XXXX_* | awk -F' ' 'BEGIN{OFS="\n"} { for(i=1; i<=NF; i++) print $i }' | parallel -j5 {} 
