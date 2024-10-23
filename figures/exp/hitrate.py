import os
import csv
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


def make_output():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_collab_cache_hit_rate(path: str, cache_policy: str):
    # Directory containing the log files
    # '../../logs/
    # <cache_policy> (e.g., LRU, FIFO)
    # /*
    # /OMP_<omp_num_thread>
    # /token_gen_time.csv'
    omp_num_thread = 1
    log_dir = os.path.join(path, cache_policy, '*', f'OMP_{omp_num_thread}',
                           '*.log')

    # Read data from log files
    df = {}
    result = {}
    for file in glob.glob(log_dir):
        # CPU_BS8_NWAYS7_logs
        cache_nblocks = int(
            os.path.basename(os.path.dirname(
                os.path.dirname(file))).split('_')[1][2:])
        cache_nways = int(
            os.path.basename(os.path.dirname(
                os.path.dirname(file))).split('_')[2][5:])
        data = []
        # fiind the number behind 'Cache hit 1: ' and 'Cache hit 2: ' from the log file
        cache_hit_1 = 0
        cache_hit_2 = 0
        with open(file, 'r') as f:
            for line in f:
                if 'Cache hit 1: ' in line:
                    cache_hit_1 = float(line.split('Cache hit 1: ')[-1])
                if 'Cache hit 2: ' in line:
                    cache_hit_2 = float(line.split('Cache hit 2: ')[-1])
        result[(cache_nblocks, cache_nways)] = [cache_hit_1, cache_hit_2]
    return result


if __name__ == '__main__':
    # Parse the argument --num-experts given from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-experts', type=int, default=8)
    args = parser.parse_args()

    # Get the cache hit rate under different cache configurations
    path = '../../logs/'
    cache_policy = ['LRU', 'FIFO']
    cache_hit_rate = {}
    for policy in cache_policy:
        cache_hit_rate[policy] = get_collab_cache_hit_rate(path, policy)
        print(
            f'Cache hit rate under {policy} policy: {cache_hit_rate[policy]}')
    cache_settings = list(cache_hit_rate[cache_policy[0]].keys())
    # sort the cache settings by the number of blocks in decending order
    cache_settings.sort(key=lambda x: x[0])
    print(f'Cache settings: {cache_settings}')

    # Draw a line chart for cache hit rate under different cache configurations and policies
    # Prepare data for plotting

    # calculate the rate for 1 expert hit under random circumstances:
    cache_hit_rate['random'] = {}
    num_exp = args.num_experts
    for setting in cache_settings:
        cache_hit_rate['random'][setting] = [
            1 - (((num_exp - setting[1]) / (num_exp)) *
                 (((num_exp - setting[1] - 1)) / (num_exp - 1))),
            (setting[1] / num_exp) * ((setting[1] - 1) / (num_exp - 1))
        ]
        # scale by 100%
        cache_hit_rate['random'][setting] = [
            rate * 100 for rate in cache_hit_rate['random'][setting]
        ]
    print(f'Cache hit rate under random policy: {cache_hit_rate["random"]}')

    cache_settings_str = [
        f'({setting[0]}, {setting[1]})' for setting in cache_settings
    ]
    cache_hit_1_LRU = [
        cache_hit_rate['LRU'][setting][0] for setting in cache_settings
    ]
    cache_hit_2_LRU = [
        cache_hit_rate['LRU'][setting][1] for setting in cache_settings
    ]
    cache_hit_1_FIFO = [
        cache_hit_rate['FIFO'][setting][0] for setting in cache_settings
    ]
    cache_hit_2_FIFO = [
        cache_hit_rate['FIFO'][setting][1] for setting in cache_settings
    ]
    cache_hit_1_random = [
        cache_hit_rate['random'][setting][0] for setting in cache_settings
    ]
    cache_hit_2_random = [
        cache_hit_rate['random'][setting][1] for setting in cache_settings
    ]
    plt.figure(figsize=(9, 3.5))
    plt.plot(cache_settings_str,
             cache_hit_1_random,
             label='expert(s) hit (Random)',
             marker='^')
    plt.plot(cache_settings_str,
             cache_hit_1_LRU,
             label='expert(s) hit (LRU)',
             marker='o')
    plt.plot(cache_settings_str,
             cache_hit_1_FIFO,
             label='expert(s) hit (FIFO)',
             marker='s')

    # Plotting
    plt.plot(cache_settings_str,
             cache_hit_2_random,
             label='2 experts hit (Random)',
             marker='^')
    plt.plot(cache_settings_str,
             cache_hit_2_LRU,
             label='2 experts hit (LRU)',
             marker='o')
    plt.plot(cache_settings_str,
             cache_hit_2_FIFO,
             label='2 experts hit (FIFO)',
             marker='s')

    plt.xlabel('Cache Settings (num_blocks, num_ways)')
    plt.ylabel('Cache Hit Rate (%)')
    plt.title(
        'Cache Hit Rate under Different Cache Configurations and Policies')
    plt.legend()
    plt.grid(True)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(make_output(), 'cache_hit_rate.pdf'),
                format='pdf')
    plt.show()
    # There are two different lines in terms of cache policies
