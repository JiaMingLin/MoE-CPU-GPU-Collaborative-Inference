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

def get_cpu_tmp_freq(path: str, omp_num_threads: int):
    # Initialize lists to store temperatures and CPU frequencies
    temperatures = []
    cpu_frequencies = []
    log = os.path.join(path, f'OMP_{omp_num_threads}', 'cpu_freq_avg.csv')

    # Read files from the specified directory
    for file in glob.glob(log):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip the header row
            for row in reader:
                temperatures.append(float(row[1]))  # Assuming temperature is in the first column
                cpu_frequencies.append(float(row[2]))  # Assuming CPU frequency is in the second column
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
    args = parser.parse_args()

    omp = args.omp
    df = get_cpu_tmp_freq('../../logs/OMP_logs', omp)
    print(df)
    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot temperature on the left y-axis
    ax1.plot(df['Temperature'], color='tab:red', label='Temperature')
    ax1.set_xlabel('Time (sec.)')
    ax1.set_ylabel('Temperature', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for CPU frequency
    ax2 = ax1.twinx()
    ax2.plot(df['CPU Frequency'], color='tab:blue', label='CPU Frequency')
    ax2.set_ylabel('CPU Frequency', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Set frequency limit
    ax2.set_ylim(4200, 5500)

    # Add a title and show the plot
    plt.title(f'Temperature and CPU Frequency Over Time (# of threads: {omp})')
    fig.tight_layout()  # Adjust layout to make room for both y-axes

    output_dir = make_output()
    plt.savefig(os.path.join(output_dir, f'temperature_OMP_{omp}.pdf'), format='pdf', bbox_inches='tight')
    plt.show()