from mldataforge.indexing import shuffle_permutation
from mldataforge.utils import load_jinx_paths

from utils import start, stop

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_jinx_paths(f"data/{tmp_dir}/{main_file}", encoding="b64" if main_file.endswith("_base64.jinx") else "a85", mmap=True)
indices= shuffle_permutation(len(ds), seed=42)

for i in range(len(ds)):
    ds[indices[i]]

stop(wall_start, cpu_start)
