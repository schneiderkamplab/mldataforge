import filecmp
from mldataforge.commands.index import index_identity, index_join, index_slice
from mldataforge.commands.join import join_mds
import pytest

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.dependency(name="indexing")
@pytest.mark.parametrize("num_slices,per_slice", [
    (1, 1),
    (1, 10),
    (1, 100),
    (10, 1),
    (10, 10),
    (10, 100),
    (100, 1000),
    (100, 10000),
    (100, 100000),
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

@pytest.mark.dependency(depends=["indexing"])
@pytest.mark.dependency(name="shuffling")
@pytest.mark.parametrize("seed,index,out_file,in_file", [
    (42, None, "test.shuffled.mds", "test.jsonl.mds"),
    (-42, None, "test.unshuffled.mds", "test.shuffled.mds"),
    (None, "test.identity.index", "test.identity.mds", "test.jsonl.mds"),
])
def test_shuffling(seed, index, out_file, in_file, tmp_dir, scale_factor):
    if index is not None:
        index = str(tmp_dir / index)
    join_mds(
        output_dir=str(tmp_dir / out_file),
        mds_directories=[str(tmp_dir / in_file)],
        compression=None,
        compression_args={"processes": 64},
        overwrite=True,
        yes=True,
        batch_size=2**10*scale_factor,
        buf_size=2**14*scale_factor,
        no_bulk=True,
        shard_size=2**26,
        no_pigz=True,
        trafo=None,
        shuffle=seed,
        index=index,
    )
    dircmp = filecmp.dircmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.mds"))
    if seed is None or seed < 0:
        assert len(dircmp.diff_files) == 0, f"Different files: {dircmp.diff_files}"
    else:
        assert len(dircmp.diff_files), f"Shuffling had no effect!"
    assert len(dircmp.left_only) == 0, f"Left only files: {dircmp.left_only}"
    assert len(dircmp.right_only) == 0, f"Right only files: {dircmp.right_only}"
    assert len(dircmp.funny_files) == 0, f"Funny files: {dircmp.funny_files}"        
