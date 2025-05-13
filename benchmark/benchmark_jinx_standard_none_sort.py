from mldataforge.indexing import sort_permutation
from mldataforge.utils import load_jinx_paths

from utils import start, stop

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_jinx_paths(f"data/{tmp_dir}/{main_file}")
indices = sort_permutation(ds, 'def key(sample): return sample["id"]')

for i in range(len(ds)):
    ds[indices[i]]

stop(wall_start, cpu_start)
