import filecmp
from mldataforge.commands.index import index_identity, index_join, index_slice
from mldataforge.commands.join import join_jinx, join_mds
from mldataforge.indexing import shuffle_permutation
from mldataforge.utils import save_jinx, save_mds
import numpy as np
import pytest

@pytest.mark.parametrize("num_slices,per_slice", [
    (1, 1),
    (1, 10),
    (1, 100),
    (10, 1),
    (10, 10),
    (10, 100),
    (100, 1000),
    (100, 10000),
    pytest.param(100, 100000, marks=pytest.mark.dependency(name="index_large", scope="session")),
])
def test_indexing(num_slices, per_slice, tmp_dir):
    # Test identity indexing
    index_identity(
        output_file=str(tmp_dir / "test.identity.index"),
        mds_directories=[],
        overwrite=True,
        yes=True,
        number=per_slice*num_slices,
        offset=0,
        every=None,
    )
    assert (tmp_dir / "test.identity.index").exists()

    # Test slice indexing
    for i in range(num_slices):
        index_slice(
            output_file=str(tmp_dir / f"test.sliced{i}.index"),
            input_file=str(tmp_dir / "test.identity.index"),
            overwrite=True,
            yes=True,
            number=per_slice,
            offset=i * per_slice,
            every=None,
        )
        assert (tmp_dir / f"test.sliced{i}.index").exists()

    # Test join indexing
    index_join(
        output_file=str(tmp_dir / "test.joined.index"),
        input_files=[str(tmp_dir / f"test.sliced{i}.index") for i in range(num_slices)],
        overwrite=True,
        yes=True,
        number=per_slice*num_slices,
        offset=0,
        every=None,
    )
    assert (tmp_dir / "test.joined.index").exists()
    assert filecmp.cmp(str(tmp_dir / "test.joined.index"), str(tmp_dir / "test.identity.index"), shallow=False), "Joined index does not match the original index!"

@pytest.mark.parametrize("fmt,seed,index,out_file,in_file", [
    pytest.param("jinx", 42, None, "test.shuffled.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="shuffle_jinx_shuffled", dependency=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", -42, None, "test.unshuffled.jinx", "test.shuffled.jinx", marks=pytest.mark.dependency(dependency=["shuffle_jinx_shuffled"], scope="session")),
    pytest.param("jinx", None, "test.identity.index", "test.identity.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(dependency=["convert_jsonl_jinx", "index_large"], scope="session")),
    pytest.param("mds", 42, None, "test.shuffled.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="shuffle_mds_shuffled", dependency=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", -42, None, "test.unshuffled.mds", "test.shuffled.mds", marks=pytest.mark.dependency(dependency=["shuffle_mds_shuffled"], scope="session")),
    pytest.param("mds", None, "test.identity.index", "test.identity.mds", "test.jsonl.mds", marks=pytest.mark.dependency(dependency=["convert_jsonl_mds", "index_large"], scope="session")),
])
def test_shuffling(fmt,seed, index, out_file, in_file, tmp_dir, scale_factor, jsonl_tools):
    if index is not None:
        index = str(tmp_dir / index)
    if fmt == "jinx":
        join_jinx(
            output_file=str(tmp_dir / out_file),
            jinx_paths=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=None,
            shuffle=seed,
            index=index,
            sort_key=None,
            lazy=True,
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=None,
        )
        if seed is None or seed < 0:
            assert jsonl_tools.equal(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.jinx")), f"Files {out_file} and test.jsonl.jinx are different"
        else:
            assert not filecmp.cmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.jinx"), shallow=False), f"Files {out_file} and test.jsonl.jinx are the same"
    elif fmt == "mds":
        join_mds(
            output_dir=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            buf_size=2**14*scale_factor,
            reader="ram",
            shard_size=2**26,
            no_pigz=True,
            trafo=None,
            shuffle=seed,
            index=index,
            sort_key=None,
        )
        dircmp = filecmp.dircmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.mds"))
        if seed is None or seed < 0:
            assert len(dircmp.diff_files) == 0, f"Different files: {dircmp.diff_files}"
        else:
            assert len(dircmp.diff_files), f"Shuffling had no effect!"
        assert len(dircmp.left_only) == 0, f"Left only files: {dircmp.left_only}"
        assert len(dircmp.right_only) == 0, f"Right only files: {dircmp.right_only}"
        assert len(dircmp.funny_files) == 0, f"Funny files: {dircmp.funny_files}"        

@pytest.mark.parametrize("fmt,param,sort_key,input_directory", [
    pytest.param("jinx", None, "def key(sample): return sample['id']", "test.tokenized.jinx", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_jinx"], scope="session")),
    pytest.param("jinx", None, "def key(sample): return len(sample['input_ids'])", "test.tokenized.jinx", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_jinx"], scope="session")),
    ("jinx", None, "def key(sample): return sample['id']", None),
    pytest.param("jinx", "snappy", "def key(sample): return sample['id']", "test.tokenized.jinx", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_jinx"], scope="session")),
    pytest.param("jinx", "snappy", "def key(sample): return len(sample['input_ids'])", "test.tokenized.jinx", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_jinx"], scope="session")),
    ("jinx", "snappy", "def key(sample): return sample['id']", None),
    pytest.param("mds", "ram", "def key(sample): return sample['id']", "test.tokenized.mds", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_mds"], scope="session")),
    pytest.param("mds", "ram", "def key(sample): return len(sample['input_ids'])", "test.tokenized.mds", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_mds"], scope="session")),
    ("mds", "ram", "def key(sample): return sample['id']", None),
    pytest.param("mds", "streaming", "def key(sample): return sample['id']", "test.tokenized.mds", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_mds"], scope="session")),
    pytest.param("mds", "streaming", "def key(sample): return len(sample['input_ids'])", "test.tokenized.mds", marks=pytest.mark.dependency(depends=["trafos_test_tokenized_mds"], scope="session")),
    ("mds", "streaming", "def key(sample): return sample['id']", None),
])
def test_sorting(fmt, param, sort_key, input_directory, tmp_dir, request, jsonl_tools):
    if input_directory is None:
        num_indices = request.config.getoption("--indices")
        def id_iterator(it):
            for i in it:
                yield {"id": int(i), "payload": np.random.randint(0, 2**32, size=2**10, dtype=np.uint64)}
        indices = shuffle_permutation(num_indices, seed=42)
        input_directory = f"test.{num_indices}.{param}.{fmt}"
        if fmt == "jinx":
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
        elif fmt == "mds":
            save_mds(
                id_iterator(indices),
                str(tmp_dir / input_directory),
                compression=None,
                compression_args={"processes": 64},
                pigz=False,
                buf_size=2**14,
                shard_size=None,
                size_hint=None,
                overwrite=True,
                yes=True,
                trafo=None,
            )
    sorted_file = str(tmp_dir / f"{input_directory}.sorted.{fmt}")
    resorted_file = str(tmp_dir / f"{input_directory}.resorted.{fmt}")
    if fmt == "jinx":
        join_jinx(
            output_file=sorted_file,
            jinx_paths=[str(tmp_dir / input_directory)],
            compression=param,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=sort_key,
            lazy=True,
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=2**8,
        )
        join_jinx(
            output_file=resorted_file,
            jinx_paths=[sorted_file],
            compression=param,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=sort_key,
            lazy=True,
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=2**8,
        )
        assert jsonl_tools.equal(sorted_file, resorted_file), f"Files {sorted_file} and {resorted_file} are different"
    elif fmt == "mds":
        join_mds(
            output_dir=sorted_file,
            mds_directories=[str(tmp_dir / input_directory)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            batch_size=2**10,
            buf_size=2**14,
            reader=param,
            shard_size=2**26,
            no_pigz=True,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=sort_key,
        )
        join_mds(
            output_dir=resorted_file,
            mds_directories=[sorted_file],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            batch_size=2**10,
            buf_size=2**14,
            reader=param,
            shard_size=2**26,
            no_pigz=True,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=sort_key,
        )
        dircmp = filecmp.dircmp(sorted_file, resorted_file)
        assert len(dircmp.diff_files) == 0, f"Different files: {dircmp.diff_files}"
        assert len(dircmp.left_only) == 0, f"Left only files: {dircmp.left_only}"
        assert len(dircmp.right_only) == 0, f"Right only files: {dircmp.right_only}"
        assert len(dircmp.funny_files) == 0, f"Funny files: {dircmp.funny_files}"        
