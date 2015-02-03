#! /bin/bash

trial="no_interaction_bit30"
modelName="model_${trial}"
resultName="plus_submission_${trial}.csv"
logName="log_${trial}.txt"
Full=`seq 1 1 10`

#Cross Validation on Dates

for fold in `seq 1 1 5`
do  
    string=''
    string_rest=''
    count=0
    count_rest=0
    for element in $Full
    do
        if [ $element -ne $((2 * $fold - 1)) -a $element -ne $((2 * $fold)) ]
        then
            if [ $element -ne 10 ]
            then
                string+="14102$element"
            else
                string+="141030"
            fi
            
            count=$((count + 1))
            if [ $count -ne 8 ]
            then
                string+=','
            fi
            echo $fold ' = ' $string
        else
            if [ $element -ne 10 ]
            then
                string_rest+="14102$element"
            else
                string_rest+="141030"
            fi
            
            count_rest=$((count_rest + 1))
            if [ $count_rest -ne 2 ]
            then
                string_rest+=','
            fi
            echo $fold ' = ' $string_rest
        fi
    done
    

    echo "pypy fast_solution_plus.py train --train data/train.csv --validate data/train.csv -o model/${modelName}_${fold} --alpha 0.11 --beta 2.0 --L1 2.0 --L2 2.0 --dropout 0.8 --bits 30 --n_epochs 10 --sparse --onlydays $string  2>> plus_log/${logName}_${fold}_train" > run_${trial}_${fold}_train.sh
    
    #echo "pypy fast_solution_plus.py predict --test data/test.csv -i model/${modelName}_${fold} -p plus_result/${resultName}_${fold} 2>> plus_log/${logName}_${fold} " >> run_${trial}_${fold}.sh

done

sed "s/run_XXXX/run_${trial}/" master_template.sh > run_master.sh
chmod 770 run_${trial}_*.sh run_master.sh
