from mldataforge.commands.join import join_jinx, join_mds
from mldataforge.indexing import shuffle_permutation
from mldataforge.utils import save_jinx, save_mds
import numpy as np
import pytest
import re

def clean(x):
    return re.sub(r'[^A-Za-z0-9._-]', '', str(x))

@pytest.mark.parametrize("fmt,param,compression,trafo", [
    ("jinx", {"lazy": True, "mmap": False}, None, ["def process(sample): return {'id': sample['id']}"]),
    ("jinx", {"lazy": True, "mmap": False}, "snappy", ["def process(sample): return {'id': sample['id']}"]),
    ("jinx", {"lazy": False, "mmap": False}, None, ["def process(sample): return {'id': sample['id']}"]),
    ("jinx", {"lazy": False, "mmap": False}, "snappy", ["def process(sample): return {'id': sample['id']}"]),
    ("jinx", {"lazy": True, "mmap": True}, None, ["def process(sample): return {'id': sample['id']}"]),
    ("jinx", {"lazy": True, "mmap": True}, "snappy", ["def process(sample): return {'id': sample['id']}"]),
    ("jinx", {"lazy": False, "mmap": True}, None, ["def process(sample): return {'id': sample['id']}"]),
    ("jinx", {"lazy": False, "mmap": True}, "snappy", ["def process(sample): return {'id': sample['id']}"]),
    ("mds", "ram", None, ["def process(sample): return {'id': sample['id']}"]),
    ("mds", "ram", "sample::snappy", ["def process(sample): return {'id': sample['id']}"]),
    ("mds", "bulk", None, ["def process(sample): return {'id': sample['id']}"]),
    ("mds", "bulk", "sample::snappy", ["def process(sample): return {'id': sample['id']}"]),
    ("mds", "streaming", None, ["def process(sample): return {'id': sample['id']}"]),
], ids=clean)
def test_lazy(fmt, param, compression, trafo, tmp_dir, request):
    num_indices = request.config.getoption("--indices")
    def id_iterator(it):
        for i in it:
            yield {"id": int(i), "payload": np.random.randint(0, 2**32, size=2**14, dtype=np.uint64)}
    indices = shuffle_permutation(num_indices, seed=42)
    input_directory = tmp_dir / f"test.{num_indices}.{compression}.{clean(param)}.{fmt}"
    if fmt == "jinx":
        save_jinx(
            id_iterator(indices),
            str(input_directory),
            compression=compression,
            compression_args={"processes": 64},
            shard_size=None,
            size_hint=None,
            overwrite=True,
            yes=True,
            trafo=None,
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=2**8,
            ext_sep=".",
        )
    elif fmt == "mds":
        save_mds(
            id_iterator(indices),
            str(input_directory),
            compression=compression,
            shard_size=2**31,
            size_hint=None,
            overwrite=True,
            yes=True,
            trafo=None,
            pigz=False,
        )
    projected_file = tmp_dir / f"test.{num_indices}.{compression}.{clean(param)}.projected.{fmt}"
    if fmt == "jinx":
        join_jinx(
            output_file=str(projected_file),
            jinx_paths=[str(input_directory)],
            compression=compression,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=trafo,
            mmap=param["mmap"],
            shuffle=None,
            index=None,
            sort_key=None,
            lazy=param["lazy"],
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=2**8,
            ext_sep=".",
        )
    elif fmt == "mds":
        join_mds(
            output_dir=str(projected_file),
            mds_directories=[str(input_directory)],
            compression=compression,
            compression_args={"processes": 64},
            batch_size=2**6,
            buf_size=2**24,
            reader=param,
            no_pigz=True,
            overwrite=True,
            yes=True,
            shard_size=2**31,
            trafo=trafo,
            shuffle=None,
            index=None,
            sort_key=None,
        )
    assert projected_file.exists(), f"File {projected_file} does not exist"
