import filecmp
from mldataforge.commands.join import join_mds
import pytest

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.dependency(name="shuffling")
@pytest.mark.parametrize("seed,out_file,in_file", [
    (42, "test.shuffled.mds", "test.jsonl.mds"),
    (-42, "test.unshuffled.mds", "test.shuffled.mds"),
])
def test_shuffling(seed, out_file, in_file, tmp_dir, scale_factor):
    join_mds(
        output_dir=str(tmp_dir / out_file),
        mds_directories=[str(tmp_dir / in_file)],
        compression=None,
        processes=64,
        overwrite=True,
        yes=True,
        batch_size=2**10*scale_factor,
        buf_size=2**14*scale_factor,
        no_bulk=True,
        shard_size=2**26,
        no_pigz=True,
        trafo=None,
        shuffle=seed,
    )
    dircmp = filecmp.dircmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.mds"))
    if seed < 0:
        assert len(dircmp.diff_files) == 0, f"Different files: {dircmp.diff_files}"
    else:
        assert len(dircmp.diff_files), f"Shuffling had no effect!"
    assert len(dircmp.left_only) == 0, f"Left only files: {dircmp.left_only}"
    assert len(dircmp.right_only) == 0, f"Right only files: {dircmp.right_only}"
    assert len(dircmp.funny_files) == 0, f"Funny files: {dircmp.funny_files}"        
