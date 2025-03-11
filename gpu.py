import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import torch
import subprocess

def get_nvidia_smi_output():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,pci.bus_id', '--format=csv,noheader']).decode()
        return [line.strip().split(', ') for line in output.split('\n') if line.strip()]
    except:
        return []

print("System GPUs from nvidia-smi:")
nvidia_gpus = get_nvidia_smi_output()
for gpu in nvidia_gpus:
    print(f"  GPU {gpu[0]}: {gpu[1]} (Bus ID: {gpu[2]})")

print("\nEnvironment variables:")
print(f"CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER', 'Not set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

print("\nPyTorch CUDA information:")
print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"\nPyTorch Device {i}:")
    print(f"  Name: {torch.cuda.get_device_name(i)}")
    prop = torch.cuda.get_device_properties(i)
    print(f"  Total memory: {prop.total_memory / 1024**2:.0f} MB")