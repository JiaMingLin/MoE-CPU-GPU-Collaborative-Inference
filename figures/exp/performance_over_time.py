import glob
import os
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def make_output():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_cpu_or_gpu_only_time(path: str, omp_num_threads: int = -1):
    # Initialize lists to store temperatures and CPU frequencies
    token_time = []
    temp = []
    if omp_num_threads == -1:
        log = os.path.join(path, 'GPU_Offload_log', 'token_gen_time.csv')
    else:
        log = os.path.join(path, 'OMP_logs', f'OMP_{omp_num_threads}',
                           'token_gen_time.csv')

    # Read files from the specified directory
    for file in glob.glob(log):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip the header row
            for row in reader:
                token_time.append(float(
                    row[1]))  # Assuming time is in the second column
                temp.append(float(row[5]))
    # Create a DataFrame from the lists
    df = pd.DataFrame({'Generation time': token_time, 'Temperature': temp})
    return df


def get_collab_token_gen_throughput(path: str, omp_num_threads: int):
    # Directory containing the log files
    # '../../logs/LRU/*/OMP_*/token_gen_time.csv'
    log_dir = os.path.join(path, '*', f'OMP_{omp_num_threads}',
                           'token_gen_time.csv')

    # Read data from log files
    df = {}
    token_time = []
    temp = []
    best_name = ''

    for file in glob.glob(log_dir):
        omp_number = int(
            os.path.basename(os.path.dirname(file)).split('_')[-1])
        data = []
        temp_data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip the header row
            for row in reader:
                # Assuming time is in the second column
                data.append(float(row[1]))
                temp_data.append(float(row[5]))
            if len(token_time) == 0:
                token_time = data
                best_name = file
                temp = temp_data
            else:
                avg_data = sum(data) / len(data)
                avg_token_time = sum(token_time) / len(token_time)
                if avg_data < avg_token_time:
                    token_time = data
                    best_name = file
                    temp = temp_data

    # Get the nth index from the DataFrame
    print(f'Best cache policy: {best_name}')
    df = pd.DataFrame({'Generation time': token_time, 'Temperature': temp})
    return df


if __name__ == '__main__':
    # argv input: --omp
    # parse argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--omp', type=int, help='Number of OpenMP threads')
    args = parser.parse_args()

    omp = args.omp

    df_cpu = get_cpu_or_gpu_only_time('../../logs', omp)
    df_gpu = get_cpu_or_gpu_only_time('../../logs')
    df_collab = get_collab_token_gen_throughput('../../logs/LRU', omp)

    # Apply a rolling mean to smooth the data
    WINDOW_SIZE = 5  # You can adjust the window size as needed
    LIMIT_ITEMS = 1800  # Limit the number of items to plot
    df_cpu['Generation time'] = df_cpu['Generation time'].rolling(
        window=WINDOW_SIZE).mean()[:LIMIT_ITEMS] * 1e3
    df_cpu['Temperature'] = df_cpu['Temperature'].rolling(
        window=WINDOW_SIZE).mean()[:LIMIT_ITEMS]
    df_gpu['Generation time'] = df_gpu['Generation time'].rolling(
        window=WINDOW_SIZE).mean()[:LIMIT_ITEMS] * 1e3
    df_collab['Temperature'] = df_collab['Temperature'].rolling(
        window=WINDOW_SIZE).mean()[:LIMIT_ITEMS]
    df_collab['Generation time'] = df_collab['Generation time'].rolling(
        window=WINDOW_SIZE).mean()[:LIMIT_ITEMS] * 1e3
    df_collab['Temperature'] = df_collab['Temperature'].rolling(
        window=WINDOW_SIZE).mean()[:LIMIT_ITEMS]

    fig, ax1 = plt.subplots(figsize=(10, 3))

    ax1.plot(df_gpu['Generation time'],
             label='GPU Offloading',
             color='tab:blue')
    ax1.plot(df_cpu['Generation time'], label='CPU Only', color='tab:orange')
    ax1.plot(df_collab['Generation time'],
             label='CPU-GPU Collab',
             color='tab:green')
    ax1.set_ylim(200, 1000)
    ax1.set_xlabel('Tokens')
    ax1.set_ylabel('Generation Time (ms.)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax1.set_title(
    #     f'Token Generation Time Comparison Over Time (# of threads: {omp})')
    # ax1.legend(loc=(0.01, 0.6))
    # ax1.legend(loc="upper")
    ax1.legend(bbox_to_anchor=(-0.07, 0.98, 1, 0.2), loc="upper left", ncol=3)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(df_cpu['Temperature'],
             label='CPU Only Temp.',
             color='#b84f23',
             linestyle='--')
    ax2.plot(df_collab['Temperature'],
             label='CPU-GPU Collab Temp.',
             color='#05754e',
             linestyle='--')
    ax2.set_ylabel('Temperature (°C)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc="upper center")
    ax2.legend(bbox_to_anchor=(0.05, 0.98, 1, 0.2), loc="upper right", ncol=2)

    output_dir = make_output()
    fig.savefig(os.path.join(output_dir, f'gen_time_history_OMP_{omp}.pdf'),
                bbox_inches='tight',
                format='pdf')
    plt.show()
