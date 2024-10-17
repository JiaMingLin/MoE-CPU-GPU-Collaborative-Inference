#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR

# Loop over # of ways from 3 to 6
for policy in "LRU" "FIFO"
    do
    echo "Processing policy: $policy"
    mkdir -p $policy
    for num_ways in 8 7 6 5 4 3 2
        do

        echo "Processing value: $i"
        # Calculate the number of ways
        block_size=$((58 / num_ways))
        echo "Processing block size: $block_size, number of ways: $num_ways"

        # Set the log base directory based on block size and number of ways
        LOG_BASE=${policy}/CPU_BS${block_size}_NWAYS${num_ways}_logs
        mkdir -p $LOG_BASE
        
        # Loop through specific numbers of threads
        for i in 24 16 8 4 2 1
            do
            
            SUB_LOG_BASE=$LOG_BASE/OMP_$i
            mkdir -p $SUB_LOG_BASE
            pushd $SUB_LOG_BASE

            # Redirect both stdout and stderr to the log file
            OMP_NUM_THREADS=$i bash ${SCRIPT_DIR}/test.sh --cache-nblocks $block_size --cache-nways $num_ways --breakdown-csv OMP_${i}_blocks_${block_size}_ways_${num_ways}.csv --cache-replace-policy ${policy} 2>&1 | tee OMP_${i}_blocks_${block_size}_ways_${num_ways}.log
            sleep 60

            popd
        done
    done
done