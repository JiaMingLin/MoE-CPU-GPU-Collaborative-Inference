import os
import csv
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



def make_output():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_cpu_token_gen_throughput(path: str, nth: int):
    # Directory containing the log files
    # '../../logs/OMP_logs/OMP_*/token_gen_time.csv'
    log_dir = path

    # Read data from log files
    df = {}
    tokenGenThroughput = []

    for file in glob.glob(log_dir):
        omp_number = int(os.path.basename(os.path.dirname(file)).split('_')[-1])
        print(f"OMP Number: {omp_number}") 
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        df[omp_number] = pd.DataFrame(data[1:], columns=data[0])

        # Get the nth index from the DataFrame
        tokenGenThroughput.append([omp_number, df[omp_number].iloc[nth-1]['Accumulated Generation Throughput (tokens/s)']])

    return tokenGenThroughput, df

def get_gpu_token_gen_throughput(path: str, nth: int):
    # Directory containing the log files
    # '../../logs/GPU_Offload_log/token_gen_time.csv'
    log_dir = path

    # Read data from log files
    df = {}
    tokenGenThroughput = []

    for file in glob.glob(log_dir):
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        df['GPU'] = pd.DataFrame(data[1:], columns=data[0])

        # Get the nth index from the DataFrame
        tokenGenThroughput.append(['GPU', df['GPU'].iloc[nth-1]['Accumulated Generation Throughput (tokens/s)']])

    return tokenGenThroughput, df
# Convert to DataFrame

nth = 512
cpu_token_throughput, df_cpu = get_cpu_token_gen_throughput('../../logs/OMP_logs/OMP_*/token_gen_time.csv', nth)
gpu_token_throughput, df_gpu = get_gpu_token_gen_throughput('../../logs/GPU_Offload_log/token_gen_time.csv', nth)

# Convert the list to a DataFrame
df_cpu_throughput = pd.DataFrame(cpu_token_throughput, columns=['OMP_NUM_THREADS', 'Throughput'])

# Convert the Throughput column to numeric
df_cpu_throughput['Throughput'] = pd.to_numeric(df_cpu_throughput['Throughput'])

# Sort by OMP_NUM_THREADS in ascending order
df_cpu_throughput = df_cpu_throughput.sort_values(by='OMP_NUM_THREADS')

# Extract GPU throughput value
gpu_throughput_value = float(gpu_token_throughput[0][1])

# Plot bar chart
plt.figure(figsize=(10, 4))
plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 1.3})
plt.bar(range(1, len(df_cpu_throughput) + 1), df_cpu_throughput['Throughput'], color='skyblue')
plt.xlabel('OMP_NUM_THREADS')
plt.ylabel('Throughput (tokens/sec.)')
plt.title('CPU Token Generation Throughput for 512 Tokens')
plt.xticks(range(1, len(df_cpu_throughput) + 1), df_cpu_throughput['OMP_NUM_THREADS'])  # Set x-axis ticks from 1 to n

# Draw horizontal line for GPU throughput
plt.axhline(y=gpu_throughput_value, color='r', linestyle='--', label=f'Throughput of GPU offloading: {gpu_throughput_value:.2f} tokens/s')
plt.legend()
# Add number labels in the center of each bar
for i, v in enumerate(df_cpu_throughput['Throughput']):
    plt.text(i + 1, min(v + 0.5, plt.ylim()[1] - 0.5), f'{v:.2f}', ha='center', va='bottom')
# Create output directory if it doesn't exist
output_dir = make_output()

plt.tight_layout()

# Save the plot in the output directory
plt.savefig(os.path.join(output_dir, 'cpu_vs_gpu_throughput.pdf'), format='pdf', bbox_inches='tight')
