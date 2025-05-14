from mldataforge.utils import load_mds_directories

from utils import start, stop

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_mds_directories([f"data/{tmp_dir}/{main_file}"], reader="ram")
for _ in ds:
    pass

stop(wall_start, cpu_start)
