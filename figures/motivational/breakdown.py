import os
import csv
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def make_output():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

def get_cpu_breakdown(path: str, nth: int):
    # Directory containing the log files
    # '../../logs/OMP_logs/OMP_*/token_gen_time.csv'
    log_dir = path

    # Read data from log files
    df = {}
    tokenGenThroughput = []
    ffn_compute = []
    attention = []
    communicate = []

    for file in glob.glob(log_dir):
        omp_number = int(os.path.basename(os.path.dirname(file)).split('_')[-1])
        print(f"OMP Number: {omp_number}") 
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        df[omp_number] = pd.DataFrame(data[1:nth-1], columns=data[0])
        # get the average of ffn_compute
        ffn_compute.append([omp_number, df[omp_number]['ffn_compute'].astype(float).mean()])
        # get the average of token_communicate
        communicate.append([omp_number, df[omp_number]['ffn_comm'].astype(float).mean()])
        # get the average of attention
        attention.append([omp_number, df[omp_number]['attn'].astype(float).mean()])

    # sort by OMP number
    ffn_compute = sorted(ffn_compute, key=lambda x: x[0])
    attention = sorted(attention, key=lambda x: x[0])
    communicate = sorted(communicate, key=lambda x: x[0])
    return ffn_compute, attention, communicate, df


def get_gpu_breakdown(path: str, nth: int):
    # Directory containing the log files
    # '../../logs/OMP_logs/OMP_*/token_gen_time.csv'
    log_dir = path

    # Read data from log files
    df = {}
    ffn_compute = []
    attention = []
    communicate = []

    for file in glob.glob(log_dir):
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        df['GPU'] = pd.DataFrame(data[1:nth-1], columns=data[0])
        # get the average of ffn_compute
        ffn_compute.append(['GPU', df['GPU']['ffn_compute'].astype(float).mean()])
        # get the average of token_communicate
        communicate.append(['GPU', df['GPU']['ffn_comm'].astype(float).mean()])
        # get the average of attention
        attention.append(['GPU', df['GPU']['attn'].astype(float).mean()])

    return ffn_compute, attention, communicate, df

ffn_compute, attention, communicate, df = get_cpu_breakdown('../../logs/OMP_logs/OMP_*/OMP.csv', 512)
gpu_ffn_compute, _, gpu_communicate, gpu_df = get_gpu_breakdown('../../logs/GPU_Offload_log/out.csv', 512)
# Prepare data for CSV
output_data = [['OMP_NUM_THREADS', 'FFN_Compute_time', 'Communication_time']]
print(gpu_ffn_compute)
output_data.append(['GPU', round(gpu_ffn_compute[0][1], 2), round(gpu_communicate[0][1], 2)])
cpu_communication_time_avg = np.mean([x[1] for x in communicate])
for i in range(len(ffn_compute)):
    output_data.append([ffn_compute[i][0], round(ffn_compute[i][1], 2), round(cpu_communication_time_avg, 2)])

# Transpose the data
output_data = list(map(list, zip(*output_data)))

make_output()
# Write data to CSV
output_file = os.path.join('output', 'breakdown.csv')
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output_data)

print(f"Data has been written to {output_file}")
print(f"FFN Compute: {ffn_compute}")
print(f"Attention: {attention}")
print(f"Token Communicate: {communicate}")