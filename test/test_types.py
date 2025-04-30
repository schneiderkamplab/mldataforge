import numpy as np
import pytest

from mldataforge.indexing import shuffle_permutation
from mldataforge.utils import load_jinx_paths, save_jinx

@pytest.mark.parametrize("fmt,data,output_path", [
    pytest.param("jinx", b"Hello World!", "raw.jinx", marks=pytest.mark.dependency(depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", np.uint64(42), "np.jinx", marks=pytest.mark.dependency(depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", np.ndarray([10,10], dtype=np.uint8), "npy.jinx", marks=pytest.mark.dependency(depends=["convert_jsonl_jinx"], scope="session")),
])
def test_types(fmt, data, output_path, tmp_dir, request):
    num_samples= request.config.getoption("--samples")
    def id_iterator(it):
        for i in it:
            yield {"id": int(i), "payload": data}
    indices = shuffle_permutation(num_samples, seed=42)
    output_path = str(tmp_dir / f"test.{num_samples}.{output_path}")
    if fmt == "jinx":
        save_jinx(
            id_iterator(indices),
            output_path,
            compression=None,
            compression_args={"processes": 64},
            shard_size=2**18,
            size_hint=None,
            overwrite=True,
            yes=True,
            trafo=None,
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=2**8,
        )
        for sample in load_jinx_paths(
            [output_path],
            split=None,
            shuffle=None,
            index=None,
            sort_key=None,
        ):
            assert type(data) == type(sample["payload"])
            if isinstance(data, np.ndarray):
                np.testing.assert_array_equal(data, sample["payload"])
            else:
                assert data == sample["payload"]
