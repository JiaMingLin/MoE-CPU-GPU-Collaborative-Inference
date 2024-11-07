import subprocess
import time
import re

def get_power_consumption():
    try:
        result = subprocess.run(['sudo', './ryzen'], capture_output=True, text=True)
        output = result.stdout
        match = re.search(r'Package:\s*(\d+\.\d+)W', output)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error: {e}")
    return None

def get_gpu_power_consumption():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        output = result.stdout.strip().split('\n')[0]
        if output:
            return float(output)
    except Exception as e:
        print(f"Error: {e}")
    return None

def main():
    total_power = 0
    count = 0
    total_gpu_power = 0
    count_gpu = 0
    start_time = time.time()

    while time.time() - start_time < 30:
        power = get_power_consumption()
        gpu_power = get_gpu_power_consumption()
        if power is not None:
            total_power += power
            print(f"CPU Power Consumption: {power:.2f} W")
            count += 1
        if gpu_power is not None:
            print(f"GPU Power Consumption: {gpu_power:.2f} W")
            total_gpu_power += gpu_power
            count_gpu += 1

        time.sleep(1)

    if count > 0:
        average_power = total_power / count
        average_gpu_power = total_gpu_power / count_gpu
        print(f"Average Power Consumption: {average_power:.2f} W")
        print(f"Average GPU Power Consumption: {average_gpu_power:.2f} W")
    else:
        print("No valid power consumption data collected.")

if __name__ == "__main__":
    main()