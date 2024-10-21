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


def get_cpu_tmp_freq(path: str):
    # Initialize lists to store temperatures and CPU frequencies
    temperatures = []
    cpu_frequencies = []
    log = os.path.join(path, 'cpu_freq_avg.csv')

    # Read files from the specified directory
    for file in glob.glob(log):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip the header row
            for row in reader:
                temperatures.append(float(
                    row[1]))  # Assuming temperature is in the first column
                cpu_frequencies.append(float(
                    row[2]))  # Assuming CPU frequency is in the second column
    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'Temperature': temperatures,
        'CPU Frequency': cpu_frequencies
    })
    return df


if __name__ == '__main__':
    # argv input: --omp
    # parse argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--omp', type=int, help='Number of OpenMP threads')
    # parser.add_argument('--collab-file',
    #                     type=str,
    #                     help='Best collab configuration to be compared with')
    args = parser.parse_args()

    omp = args.omp
    WINDOW_SIZE = 5  # You can adjust the window size as needed
    df1 = get_cpu_tmp_freq(os.path.join('../../logs/OMP_logs', f'OMP_{omp}'))
    df1 = df1.rolling(window=WINDOW_SIZE).mean()
    # df2 = get_cpu_tmp_freq(args.collab_file)
    # df2 = df2.rolling(window=WINDOW_SIZE).mean()
    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot temperature on the left y-axis
    ax1.plot(df1['Temperature'],
             color='tab:red',
             label='Temperature (CPU only)')
    # ax1.plot(df2['Temperature'],
    #          color='#7b1818',
    #          label='Temperature (CPU-GPU Collab)')
    ax1.set_xlabel('Time (sec.)')
    ax1.set_ylabel('Temperature', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for CPU frequency
    ax2 = ax1.twinx()
    ax2.plot(df1['CPU Frequency'], color='tab:blue', label='CPU Frequency')
    ax2.set_ylabel('CPU Frequency', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Set frequency limit
    ax2.set_ylim(4200, 5500)

    # Add a title and show the plot
    plt.title(f'Temperature and CPU Frequency Over Time (# of threads: {omp})')
    fig.tight_layout()  # Adjust layout to make room for both y-axes

    output_dir = make_output()
    plt.savefig(os.path.join(output_dir, f'temperature_OMP_{omp}.pdf'),
                format='pdf',
                bbox_inches='tight')
    plt.show()
