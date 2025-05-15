from utils import start, stop

from mldataforge.utils import load_jinx_paths

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_jinx_paths(f"data/{tmp_dir}/{main_file}")
for _ in ds:
    pass

stop(wall_start, cpu_start)
