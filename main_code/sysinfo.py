import torch
import platform
import psutil
import subprocess

print("===== Hardware Information =====")

# ----- CPU Name -----
cpu_name = ""

# Method 1: /proc/cpuinfo (most reliable on Linux)
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "model name" in line:
                cpu_name = line.strip().split(": ")[1]
                break
except:
    pass

# Method 2: lscpu (if the above fails)
if cpu_name == "":
    try:
        output = subprocess.check_output("lscpu", shell=True).decode()
        for line in output.split("\n"):
            if "Model name" in line:
                cpu_name = line.split(":")[1].strip()
                break
    except:
        pass

# Method 3: platform.processor() as last fallback
if cpu_name == "":
    cpu_name = platform.processor()

print("CPU:", cpu_name if cpu_name != "" else "Unknown CPU")
print("CPU Cores (logical):", psutil.cpu_count(logical=True))
print("CPU Cores (physical):", psutil.cpu_count(logical=False))

# ----- RAM -----
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"RAM: {ram_gb:.2f} GB")

# ----- GPU info -----
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        print(f"GPU {i} Memory:", torch.cuda.get_device_properties(i).total_memory // (1024**2), "MB")
else:
    print("No GPU available.")

# ----- Operating System -----
print("Operating System:", platform.platform())

print("\n===== Software Information =====")

# Python
print("Python version:", platform.python_version())

# PyTorch & CUDA
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by PyTorch:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

# NVIDIA SMI
try:
    nvidia_output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
    print("\nNVIDIA SMI Info:\n", nvidia_output)
except Exception as e:
    print("Could not run nvidia-smi:", e)

print("===== Installed Libraries =====")
try:
    import numpy, cv2, torchvision, sklearn
    print("Numpy:", numpy.__version__)
    print("OpenCV:", cv2.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Scikit-learn:", sklearn.__version__)
except:
    print("Some libraries are not installed.")
