import filecmp
from mldataforge.commands.join import join_jinx, join_jsonl, join_mds, join_parquet
from pathlib import Path
import pytest

@pytest.mark.parametrize("fmt,trafo,out_file,in_file", [
    pytest.param("jinx", ["from mldataforge.trafos import identity as process"], "test.identity.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", ["from mldataforge.trafos import flatten_json as process"], "test.flattened.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="trafos_flatten_jinx", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", ["from mldataforge.trafos import unflatten_json as process"], "test.unflattened.jinx", "test.flattened.jinx", marks=pytest.mark.dependency(depends=["trafos_flatten_jinx"], scope="session")),
    ("jsonl", ["from mldataforge.trafos import identity as process"], "test.identity.jsonl", "test.jsonl"),
    pytest.param("jsonl", ["from mldataforge.trafos import flatten_json as process"], "test.flattened.jsonl", "test.jsonl", marks=pytest.mark.dependency(name="trafos_flatten_jsonl", scope="session")),
    pytest.param("jsonl", ["from mldataforge.trafos import unflatten_json as process"], "test.unflattened.jsonl", "test.flattened.jsonl", marks=pytest.mark.dependency(depends=["trafos_flatten_jsonl"], scope="session")),
    pytest.param("mds", ["from mldataforge.trafos import identity as process"], "test.identity.mds", "test.jsonl.mds", marks=pytest.mark.dependency(depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", ["from mldataforge.trafos import flatten_json as process"], "test.flattened.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="trafos_flatten_mds", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", ["from mldataforge.trafos import unflatten_json as process"], "test.unflattened.mds", "test.flattened.mds", marks=pytest.mark.dependency(depends=["trafos_flatten_mds"], scope="session")),
    pytest.param("parquet", ["from mldataforge.trafos import identity as process"], "test.identity.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", ["from mldataforge.trafos import flatten_json as process"], "test.flattened.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(name="trafos_flatten_parquet", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", ["from mldataforge.trafos import unflatten_json as process"], "test.unflattened.parquet", "test.flattened.parquet", marks=pytest.mark.dependency(depends=["trafos_flatten_parquet"], scope="session")),
    pytest.param("jinx", [str(Path(__file__).parent / "trafo_tokenize.py")], "test.tokenized.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="trafos_test_tokenized_jinx", depends=["convert_jsonl_jinx"], scope="session")),
    ("jsonl", [str(Path(__file__).parent / "trafo_tokenize.py")], "test.tokenized.jsonl", "test.jsonl"),
    pytest.param("mds", [str(Path(__file__).parent / "trafo_tokenize.py")], "test.tokenized.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="trafos_test_tokenized_mds", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("parquet", [str(Path(__file__).parent / "trafo_tokenize.py")], "test.tokenized.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(depends=["convert_jsonl_parquet"], scope="session")),
])
def test_trafos(fmt, trafo, out_file, in_file, tmp_dir, scale_factor, request):
    if fmt == "jinx":
        join_jinx(
            output_file=str(tmp_dir / out_file),
            jinx_paths=[str(tmp_dir / in_file)],
            compression="zstd",
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=trafo,
            mmap=False,
            shuffle=None,
            index=None,
            sort_key=None,
            lazy=False,
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=2**8,
            ext_sep=".",
        )
        if "unflatten_json" in trafo or "idenitity" in trafo:
            assert filecmp.cmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.jinx"), shallow=False), f"Files {out_file} and test.jsonl.jinx are different"
    elif fmt == "jsonl":
        join_jsonl(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression="infer",
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            trafo=trafo,
        )
        if "unflatten_json" in trafo or "idenitity" in trafo:
            assert filecmp.cmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl"), shallow=False), f"Files {out_file} and test.jsonl are different"
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
            trafo=trafo,
            shuffle=None,
            index=None,
            sort_key=None,
        )
        if "unflatten_json" in trafo or "idenitity" in trafo:
            dircmp = filecmp.dircmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.mds"))
            assert len(dircmp.left_only) == 0, f"Left only files: {dircmp.left_only}"
            assert len(dircmp.right_only) == 0, f"Right only files: {dircmp.right_only}"
            assert len(dircmp.diff_files) == 0, f"Different files: {dircmp.diff_files}"
            assert len(dircmp.funny_files) == 0, f"Funny files: {dircmp.funny_files}"
    elif fmt == "parquet":
        join_parquet(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression="snappy",
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            batch_size=2**16,
            trafo=trafo,
        )
        if "unflatten_json" in trafo or "idenitity" in trafo:
            assert filecmp.cmp(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl.parquet"), shallow=False), f"Files {out_file} and test.jsonl.parquet are different"
