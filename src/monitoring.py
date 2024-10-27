import csv
import os
import subprocess
import threading
import time

from statistics import mean

LOGS = {"attn": [], "ffn_compute": [], "ffn_comm": []} # perf_analysis
EXPERT_CHOICES = {"choice": [], "cnt": []}
CUR_TOKEN_CHOICES = []

class CPUMonitor:

    def __init__(self):
        self.running = False
        self.data = []
        self.data_avg = []
        self.OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", 24)
        self.cnt = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_mhz)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def reset(self):
        self.data = []
        self.data_avg = []
        self.cnt = 0

    def get_cpu_temperature(self):
        result = subprocess.run("sensors",
                                shell=True,
                                capture_output=True,
                                text=True)
        output = result.stdout.strip().split('\n')

        temperatures = {}
        for line in output:
            if 'Tctl' in line or 'Tccd' in line:
                parts = line.split(':')
                label = parts[0].strip()
                temp = parts[1].strip().split(' ')[0].replace('+', '').replace(
                    '°C', '')
                temperatures[label] = float(temp)

        return temperatures

    def _capture_mhz(self):
        while self.running:
            result = subprocess.run(
                f"cat /proc/cpuinfo | grep 'MHz' | sed 's/cpu MHz[[:space:]]*:[[:space:]]*//' | sort -n | tail -n {self.OMP_NUM_THREADS}",
                shell=True,
                capture_output=True,
                text=True)
            # convert to int
            arr = [int(float(x)) for x in result.stdout.strip().split('\n')]
            temp = self.get_cpu_temperature()['Tctl']
            if self.cnt % 30 == 0:
                print([self.cnt, temp] + arr)
            self.data.append([self.cnt, temp] + arr)
            self.data_avg.append([self.cnt, temp, int(mean(arr))])
            self.cnt += 1
            time.sleep(1)

    def save_data(self, path, type="avg"):
        # save data as csv
        data = self.data_avg if type == "avg" else self.data
        title = ["Seconds", "Temperature"
                 ] + [f"cpu{i}" for i in range(len(data[0]))]
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(title)
            writer.writerows(data)

    def get_data_avg(self, type="avg"):
        return self.data_avg


def reset_logs():  # perf_analysis
    global LOGS
    global EXPERT_CHOICES
    global CUR_TOKEN_CHOICES
    LOGS = {"attn": [], "ffn_compute": [], "ffn_comm": []}
    EXPERT_CHOICES = {"choice": [], "cnt": []}
    CUR_TOKEN_CHOICES = []

def dump_expert_choices_to_csv(expert_choices: list, file_path: str):
    """
    Dumps the EXPERT_CHOICES list to a CSV file.

    Args:
        expert_choices (list): The list containing expert choices data.
        file_path (str): The path to the CSV file where expert choices will be dumped.
    """
    # Open the file and write the expert choices
    expert_choices = expert_choices["choice"]
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        header = ["Index"] + [f"Block_{i//2}" for i in range(64)]
        writer.writerow(header)

        # Write the expert choices
        for i, choice in enumerate(expert_choices):
            row = [i] + choice
            writer.writerow(row)


def dump_expert_cache_to_csv(expert_choices: list, file_path: str,
                             cache_size: int):
    # Open the file and write the expert choices
    expert_choices = expert_choices["cnt"]
    cache_hit_1 = 0
    cache_hit_2 = 0
    total_test = 0
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        header = ["Index"] + [f"Block_{i}" for i in range(32)]
        writer.writerow(header)

        # Write the expert choices
        for i, choice in enumerate(expert_choices):

            for idx, j in enumerate(choice):
                if idx >= cache_size:
                    break
                if j == 1:
                    cache_hit_1 += 1
                elif j == 2:
                    cache_hit_1 += 1
                    cache_hit_2 += 1
                total_test += 1
            row = [i] + choice
            writer.writerow(row)

    if total_test == 0:
        print("No cache hit due to zero cache size")
    else:
        cache_hit_1 = cache_hit_1 / total_test * 100
        cache_hit_2 = cache_hit_2 / total_test * 100
        print(
            f"Cache hit 1: {cache_hit_1:.1f}\n Cache hit 2: {cache_hit_2:.1f}")


def dump_token_generation_time_to_csv(logs: list, cpu_freq_avg: list,
                                      file_path: str):
    """
    Dumps the LOGS dictionary to a CSV file.

    Args:
        logs (list): The list containing token generation time data.
        file_path (str): The path to the CSV file where token generation time will be dumped.
    """
    # Open the file and write the logs
    with open(file_path, mode='w') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow([
            "Index", "Time", "Timestamp",
            "Accumulated Generation Throughput (tokens/s)", "CPU Frequency",
            "CPU Temperature"
        ])

        # Write the log values
        time_sum = 0
        for i, log in enumerate(logs):
            time_sum += log
            timestamp = int(time_sum)
            if timestamp >= len(cpu_freq_avg):
                cpu_freq = cpu_freq_avg[-1]
            else:
                cpu_freq = cpu_freq_avg[timestamp]
            writer.writerow([
                i + 1, log, timestamp, f"{(i+1) / time_sum:.3f}", cpu_freq[-1],
                cpu_freq[-2]
            ])


def dump_logs_to_csv(logs: dict, file_path: str):
    """
    Dumps the LOGS dictionary to a CSV file.

    Args:
        logs (dict): The dictionary containing log data.
        file_path (str): The path to the CSV file where logs will be dumped.
    """
    # Determine the maximum length of the log lists
    max_length = max(len(values) for values in logs.values())

    # Open the file and write the logs
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Index"] + list(logs.keys()))

        # Write the log values
        for i in range(max_length):
            row = [i]
            for key in logs.keys():
                # Append the value if it exists, otherwise append an empty string
                row.append(logs[key][i] if i < len(logs[key]) else "")
            writer.writerow(row)
