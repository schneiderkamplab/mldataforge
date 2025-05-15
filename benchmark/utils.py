import sys
print("Importing", file=sys.stderr)
import os
import platform
import resource
import shutil
import time

def start():
    print("Preparing", file=sys.stderr)
    tmp_dir = sys.argv[0].split("/")[-1].split(".")[0]
    if os.path.exists(f"data/{tmp_dir}"):
        shutil.rmtree(f"data/{tmp_dir}")
    os.makedirs(f"data/{tmp_dir}", exist_ok=False)
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            shutil.copytree(arg, f"data/{tmp_dir}/{arg.split('/')[-1]}")
        elif os.path.isfile(arg):
            shutil.copy(arg, f"data/{tmp_dir}")
        else:
            raise ValueError(f"Invalid argument: {arg}. Must be a file or directory.")
    main_file = sys.argv[1].split('/')[-1]
    print("Starting", file=sys.stderr)
    wall_start = time.time()
    cpu_start = os.times()
    return tmp_dir, main_file, wall_start, cpu_start

def stop(wall_start, cpu_start):
    wall_end = time.time()
    cpu_end = os.times()
    print("Stopped", file=sys.stderr)
    wall_time = wall_end - wall_start
    user_time = cpu_end.user - cpu_start.user
    system_time = cpu_end.system - cpu_start.system
    usage = resource.getrusage(resource.RUSAGE_SELF)
    ru_maxrss = usage.ru_maxrss
    print(f"Wall time: {wall_time:.6f} seconds")
    print(f"User CPU time: {user_time:.6f} seconds")
    print(f"System CPU time: {system_time:.6f} seconds")
    system = platform.system()
    if system == "Linux":
        # ru_maxrss is in kilobytes
        peak_mb = ru_maxrss / 1024
    elif system == "Darwin":  # macOS
        # ru_maxrss is in bytes
        peak_mb = ru_maxrss / 1024 / 1024
    else:
        raise ValueError(f"Unsupported platform: {system}")
    print(f"Peak memory usage: {peak_mb:.2f} MB")
