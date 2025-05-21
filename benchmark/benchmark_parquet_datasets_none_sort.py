from utils import start, stop

from mldataforge.indexing import sort_permutation
from mldataforge.utils import load_parquet_files

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_parquet_files([f"data/{tmp_dir}/{main_file}"], sort_key="def key(sample): return len(sample['messages'])")

for _ in ds:
    pass

stop(wall_start, cpu_start)
