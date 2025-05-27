from mldataforge.indexing import sort_permutation
from mldataforge.utils import load_msgpack_files

from utils import start, stop

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_msgpack_files([f"data/{tmp_dir}/{main_file}"])
indices = sort_permutation(ds, "def key(sample): return len(sample['messages'])")

for i in range(len(ds)):
    ds[int(indices[i])]

stop(wall_start, cpu_start)
