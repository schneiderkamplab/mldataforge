from utils import start, stop

from mldataforge.utils import load_jinx_paths

tmp_dir, main_file, wall_start, cpu_start = start()

ds = load_jinx_paths(f"data/{tmp_dir}/{main_file}", lazy=True, encoding="b64" if main_file.endswith("_base64.jinx") else "a85", mmap=True)
for _ in ds:
    pass

stop(wall_start, cpu_start)
