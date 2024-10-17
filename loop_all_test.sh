#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR
LOG_BASE=cpu_long_29_2_logs
mkdir -p $LOG_BASE
# Loop through specific numbers
for i in 1 2 4 8 16 24
do
    echo "Processing value: $i"
    SUB_LOG_BASE=$LOG_BASE/OMP_$i
    mkdir -p $SUB_LOG_BASE
    pushd $SUB_LOG_BASE
    OMP_NUM_THREADS=$i bash ${SCRIPT_DIR}/test.sh --cache-nblocks 29 --cache-nways 2 --breakdown-csv OMP_$i.csv | tee OMP_$i.log
    sleep 60
    # Add your custom commands here, using $i
    popd
done

