import filecmp
from mldataforge.commands.join import join_jsonl, join_mds, join_parquet
import pytest

@pytest.mark.parametrize("fmt,trafo,out_file,in_file", [
    ("jsonl", "flatten_json", "test.flattened.jsonl", "test.none.jsonl"),
    ("jsonl", "unflatten_json", "test.unflattened.jsonl", "test.flattened.jsonl"),
    ("mds", "flatten_json", "test.flattened.mds", "test.parquet.mds"),
    ("mds", "unflatten_json", "test.unflattened.mds", "test.flattened.mds"),
    ("parquet", "flatten_json", "test.flattened.parquet", "test.parquet"),
    ("parquet", "unflatten_json", "test.unflattened.parquet", "test.flattened.parquet"),
])
def test_trafos(fmt, trafo, out_file, in_file, tmp_dir):
    if fmt == "jsonl":
        join_jsonl(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression="infer",
            processes=64,
            overwrite=True,
            yes=True,
            trafo=trafo,
        )
        if trafo == "unflatten_json":
            assert filecmp.cmp(
                str(tmp_dir / out_file),
                str(tmp_dir / "test.none.jsonl"),
                shallow=False,
            ), f"Files {out_file} and test.parquet.jsonl.gz are different"
    elif fmt == "mds":
        join_mds(
            output_dir=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression=None,
            processes=64,
            overwrite=True,
            yes=True,
            batch_size=2**10,
            buf_size=2**14,
            no_bulk=False,
            shard_size=2**26,
            no_pigz=True,
            trafo=trafo,
        )
        if trafo == "unflatten_json":
            dircmp = filecmp.dircmp(str(tmp_dir / out_file), str(tmp_dir / "test.parquet.mds"))
            assert len(dircmp.left_only) == 0, f"Left only files: {dircmp.left_only}"
            assert len(dircmp.right_only) == 0, f"Right only files: {dircmp.right_only}"
            assert len(dircmp.diff_files) == 0, f"Different files: {dircmp.diff_files}"
            assert len(dircmp.funny_files) == 0, f"Funny files: {dircmp.funny_files}"
    elif fmt == "parquet":
        join_parquet(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression="snappy",
            overwrite=True,
            yes=True,
            batch_size=2**10,
            trafo=trafo,
        )
        if trafo == "unflatten_json":
            assert filecmp.cmp(
                str(tmp_dir / out_file),
                str(tmp_dir / "test.parquet"),
                shallow=False,
            ), f"Files {out_file} and test.parquet are different"
