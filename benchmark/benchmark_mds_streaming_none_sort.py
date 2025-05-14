from mldataforge.indexing import sort_permutation
from mldataforge.utils import load_mds_directories

from utils import start, stop

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_mds_directories([f"data/{tmp_dir}/{main_file}"], reader="streaming")
indices = sort_permutation(ds, "def key(sample): return len(sample['messages'])")

for i in range(len(ds)):
    ds[indices[i]]

stop(wall_start, cpu_start)
