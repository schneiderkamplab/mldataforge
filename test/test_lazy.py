from mldataforge.commands.join import join_jinx
from mldataforge.indexing import shuffle_permutation
from mldataforge.utils import save_jinx
import numpy as np
import pytest

@pytest.mark.parametrize("param,trafo", [
    (None, ["def process(sample): return sample['id']"]),
    ("snappy", ["def process(sample): return sample['id']"]),
])
def test_lazy(param, trafo, tmp_dir, request):
    num_indices = request.config.getoption("--indices")
    def id_iterator(it):
        for i in it:
            yield {"id": int(i), "payload": np.random.randint(0, 2**32, size=2**14, dtype=np.uint64)}
    indices = shuffle_permutation(num_indices, seed=42)
    input_directory = f"test.{num_indices}.{param}"
    save_jinx(
        id_iterator(indices),
        str(tmp_dir / input_directory),
        compression=param,
        compression_args={"processes": 64},
        shard_size=None,
        size_hint=None,
        overwrite=True,
        yes=True,
        trafo=None,
        compress_threshold=2**6,
        compress_ratio=1.0,
        binary_threshold=2**8,
    )
    projected_file = tmp_dir / f"{input_directory}.projected"
    join_jinx(
        output_file=str(projected_file),
        jinx_paths=[str(tmp_dir / input_directory)],
        compression=param,
        compression_args={"processes": 64},
        overwrite=True,
        yes=True,
        shard_size=None,
        trafo=trafo,
        shuffle=None,
        index=None,
        sort_key=None,
        compress_threshold=2**6,
        compress_ratio=1.0,
        binary_threshold=2**8,
    )
    assert projected_file.exists(), f"File {projected_file} does not exist"
