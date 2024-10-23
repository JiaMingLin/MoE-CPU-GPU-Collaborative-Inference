import os
import csv
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth


def make_output():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)


# Get GPU token generation throughput
# path: path to the log files
# nth: index of the throughput value to retrieve
# return: token generation throughput
def get_gpu_token_gen_throughput(path: str, nth: int):
    # Directory containing the log files
    # '../../logs/GPU_Offload_log/
    log_dir = os.path.join(path, 'token_gen_time.csv')

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
        tokenGenThroughput.append([
            'GPU',
            df['GPU'].iloc[nth -
                           1]['Accumulated Generation Throughput (tokens/s)']
        ])

    return float(tokenGenThroughput[0][1])


def get_cpuonly_token_gen_throughput(path: str, nth: int):
    # Directory containing the log files
    # '../../logs/OMP_logs/OMP_*/token_gen_time.csv'
    log_dir = path

    # Read data from log files
    df = {}
    tokenGenThroughput = []

    for file in glob.glob(log_dir):
        omp_number = int(
            os.path.basename(os.path.dirname(file)).split('_')[-1])
        print(f"OMP Number: {omp_number}")
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        df[omp_number] = pd.DataFrame(data[1:], columns=data[0])

        # Get the nth index from the DataFrame
        tokenGenThroughput.append([
            omp_number, df[omp_number].iloc[nth - 1]
            ['Accumulated Generation Throughput (tokens/s)']
        ])

    # sort by OMP number
    tokenGenThroughput = sorted(tokenGenThroughput, key=lambda x: x[0])
    return tokenGenThroughput, df


# Get CPU token generation throughput under difference cache configurations
# path: path to the root directore of the log files
# omp_num_thread: number of OpenMP threads
# cache_policy: cache replacement policy (e.g., LRU, FIFO)
# nth: index of the throughput value to retrieve
# return: token generation throughput
def get_cpu_token_gen_throughput(path: str, omp_num_thread: int,
                                 cache_policy: str, nth: int):
    # Directory containing the log files
    # '../../logs/
    # <cache_policy> (e.g., LRU, FIFO)
    # /*
    # /OMP_<omp_num_thread>
    # /token_gen_time.csv'
    log_dir = os.path.join(path, cache_policy, '*', f'OMP_{omp_num_thread}',
                           'token_gen_time.csv')

    # Read data from log files
    df = {}
    tokenGenThroughput = []

    for file in glob.glob(log_dir):
        # CPU_BS8_NWAYS7_logs
        cache_nblocks = int(
            os.path.basename(os.path.dirname(
                os.path.dirname(file))).split('_')[1][2:])
        cache_nways = int(
            os.path.basename(os.path.dirname(
                os.path.dirname(file))).split('_')[2][5:])
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        df[cache_nways] = pd.DataFrame(data[1:nth + 1], columns=data[0])

        # Get the nth index from the DataFrame
        idx = min(nth - 1, len(df[cache_nways]) - 1)
        # print(f"Index: {idx}, {file}, len: {len(df[cache_nways])}")
        tokenGenThroughput.append([
            (cache_nblocks, cache_nways), df[cache_nways].iloc[idx]
            ['Accumulated Generation Throughput (tokens/s)']
        ])

    return tokenGenThroughput, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nth', type=int, default=128)
    args = parser.parse_args()
    nth = args.nth
    log_base_dir = '../../logs/'
    omp_num_thread_list = [1, 2, 4, 8, 16, 24]
    cache_policy_list = ['LRU', 'FIFO']

    # GPU token generation throughput
    gpu_token_throughput = get_gpu_token_gen_throughput(
        '../../logs/GPU_Offload_log/', nth)

    # Store the throughput DataFrame for each cache policy
    df_obj = {}
    for omp_num_thread in omp_num_thread_list:
        df_obj[omp_num_thread] = {}
    for cache_policy in cache_policy_list:
        for omp_num_thread in omp_num_thread_list:
            df_obj[omp_num_thread][cache_policy] = None

    for omp_num_thread in omp_num_thread_list:
        for cache_policy in cache_policy_list:
            cpu_token_throughput, df_cpu = get_cpu_token_gen_throughput(
                log_base_dir, omp_num_thread, cache_policy, nth)
            df_obj[omp_num_thread][cache_policy] = cpu_token_throughput

    cpuonly_token_throughput, df_cpuonly = get_cpuonly_token_gen_throughput(
        '../../logs/OMP_logs/OMP_*/token_gen_time.csv', nth)
    # Prepare the data for plotting
    data = []

    for omp_num_thread in omp_num_thread_list:
        for cache_policy in cache_policy_list:
            for (nblocks, throughput) in df_obj[omp_num_thread][cache_policy]:
                data.append(
                    (omp_num_thread, cache_policy, nblocks, throughput))

    # Convert to DataFrame
    df_plot = pd.DataFrame(data,
                           columns=[
                               'OMP_NUM_THREADS', 'Cache_Policy',
                               '(Nblocks,Nways)', 'Throughput'
                           ])
    # Extract the number of Nblocks from the '(Nblocks,Nways)' column
    df_plot['Nblocks'] = df_plot['(Nblocks,Nways)'].apply(lambda x: x[0])

    # Sort the DataFrame
    df_plot = df_plot.sort_values(
        by=['OMP_NUM_THREADS', 'Cache_Policy', 'Nblocks'])

    # Drop the temporary 'Nblocks' column if not needed
    df_plot = df_plot.drop(columns=['Nblocks'])

    # Plotting
    # use greyscale seaborn
    fig, ax = plt.subplots(figsize=(20, 7))

    # increase font size
    plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 1.3})

    # Create a bar chart
    bar_width = 0.9
    positions = np.arange(len(df_plot['OMP_NUM_THREADS'].unique()))
    pos_len = len(df_plot[df_plot['Cache_Policy'] == cache_policy_list[0]]
                  ['(Nblocks,Nways)'].unique()) + 1

    # Define a color palette
    colors = plt.cm.Spectral(
        np.linspace(0, 1,
                    len(df_plot['(Nblocks,Nways)'].unique()) + 1))

    for i, cache_policy in enumerate(["LRU"]):
        subset = df_plot[df_plot['Cache_Policy'] == cache_policy]

        tmp = []
        for xx in cpuonly_token_throughput:
            tmp.append(float(xx[1]))
        x_position = positions + i * (bar_width) - (bar_width * (0.5 - 0.06))

        ax.bar(
            x_position,
            tmp,
            width=bar_width / pos_len,
            color=colors[0],
            edgecolor='black',  # Add border color
            label=f'CPU only')
        # Show the value on top of the bar
        for x, y in zip(x_position, tmp):
            ax.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=10)
            # label=f'{cache_policy} - {nblocks}')

        for j, nblocks in enumerate(subset['(Nblocks,Nways)'].unique()):
            jj = j + 1
            subsubset = subset[subset['(Nblocks,Nways)'] == nblocks]
            tmp = []
            for xx in subsubset['Throughput']:
                tmp.append(float(xx))
            x_position = positions + i * (
                bar_width) + jj * bar_width / pos_len - (bar_width *
                                                         (0.5 - 0.06))
            ax.bar(
                x_position,
                tmp,
                width=bar_width / pos_len,
                color=colors[jj % len(colors)],
                edgecolor='black',  # Add border color
                label=f'Cache: {nblocks}')
            # Show the value on top of the bar
            for x, y in zip(x_position, tmp):
                ax.text(x,
                        y,
                        f'{y:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=10)
                # label=f'{cache_policy} - {nblocks}')

    fs = 15
    # Draw a horizontal line for the GPU throughput
    ax.axhline(y=round(gpu_token_throughput, 2),
               color='r',
               linestyle='--',
               label='GPU Offloading')

    # Add labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(omp_num_thread_list, fontsize=fs)
    ax.set_xlabel('OMP_NUM_THREADS', fontsize=fs)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=fs)
    ax.set_title(
        'Token Generation Throughput by OMP_NUM_THREADS, Cache NBlocks',
        fontsize=fs)
    ax.legend(title='Cache: (Nblocks,Nways)', fontsize=fs)
    # Set y-axis range
    ax.set_ylim(0, 5)

    # Set the label of y-axis by 0.5 each
    ax.set_yticks(np.arange(0, 5.5, 0.5))
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs)

    # Show the plot
    # plt.show()
    # Save the plot
    make_output()
    plt.savefig(os.path.join('output', f'cpu_vs_gpu_throughput_{nth}.pdf'),
                format='pdf',
                bbox_inches='tight',
                dpi=600)

    print(f'CPU Token Gen Throughput: {cpu_token_throughput}')
    print(f'GPU Token Gen Throughput: {gpu_token_throughput}')
