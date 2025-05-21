from mldataforge.utils import load_msgpack_files

from utils import start, stop

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_msgpack_files([f"data/{tmp_dir}/{main_file}"])
for _ in ds:
    pass

stop(wall_start, cpu_start)
