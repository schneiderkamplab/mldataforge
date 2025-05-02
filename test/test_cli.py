from click.testing import CliRunner
from mldataforge.commands import cli
import pytest

@pytest.mark.parametrize("src_fmt,target_fmt,out_file,in_file", [
    pytest.param("jinx", "jsonl", "test.jsonl.jinx.jsonl", "test.jsonl.jinx", marks=pytest.mark.dependency(name="convert_jinx_jsonl", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "mds", "test.jsonl.jinx.mds", "test.jsonl.jinx", marks=pytest.mark.dependency(name="convert_jinx_mds", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "msgpack", "test.jsonl.jinx.msgpack", "test.jsonl.jinx", marks=pytest.mark.dependency(name="convert_jinx_msgpack", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "parquet", "test.jsonl.jinx.parquet", "test.jsonl.jinx", marks=pytest.mark.dependency(name="convert_jinx_parquet", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jsonl", "jinx", "test.jsonl.jinx", "test.jsonl", marks=pytest.mark.dependency(name="convert_jsonl_jinx", scope="session")),
    pytest.param("jsonl", "mds", "test.jsonl.mds", "test.jsonl", marks=pytest.mark.dependency(name="convert_jsonl_mds", scope="session")),
    pytest.param("jsonl", "parquet", "test.jsonl.parquet", "test.jsonl", marks=pytest.mark.dependency(name="convert_jsonl_parquet", scope="session")),
    pytest.param("jsonl", "msgpack", "test.jsonl.msgpack", "test.jsonl", marks=pytest.mark.dependency(name="convert_jsonl_msgpack", scope="session")),
    pytest.param("mds", "jinx", "test.jsonl.mds.jinx", "test.jsonl.mds", marks=pytest.mark.dependency(name="convert_mds_jinx", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "jsonl", "test.jsonl.mds.jsonl", "test.jsonl.mds", marks=pytest.mark.dependency(name="convert_mds_jsonl", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "parquet", "test.jsonl.mds.parquet", "test.jsonl.mds", marks=pytest.mark.dependency(name="convert_mds_parquet", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "msgpack", "test.jsonl.mds.msgpack", "test.jsonl.mds", marks=pytest.mark.dependency(name="convert_mds_msgpack", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("msgpack", "jinx", "test.jsonl.msgpack.jinx", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="convert_msgpack_jinx", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "jsonl", "test.jsonl.msgpack.jsonl", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="convert_msgpack_jsonl", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "mds", "test.jsonl.msgpack.mds", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="convert_msgpack_mds", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "parquet", "test.jsonl.msgpack.parquet", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="convert_msgpack_parquet", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("parquet", "jinx", "test.jsonl.parquet.jinx", "test.jsonl.parquet", marks=pytest.mark.dependency(name="convert_parquet_jinx", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", "jsonl", "test.jsonl.parquet.jsonl", "test.jsonl.parquet", marks=pytest.mark.dependency(name="convert_parquet_jsonl", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", "mds", "test.jsonl.parquet.mds", "test.jsonl.parquet", marks=pytest.mark.dependency(name="convert_parquet_mds", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", "msgpack", "test.jsonl.parquet.msgpack", "test.jsonl.parquet", marks=pytest.mark.dependency(name="convert_parquet_msgpack", depends=["convert_jsonl_parquet"], scope="session")),
])
def test_conversion(src_fmt, target_fmt, out_file, in_file, tmp_dir, jsonl_tools):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "convert",
        src_fmt,
        target_fmt,
        str(tmp_dir / out_file),
        str(tmp_dir / in_file),
        "--overwrite",
        "--yes"
    ])
    assert result.exit_code == 0, f"Failed conversion {src_fmt} -> {target_fmt}: {result.output}"
    assert (tmp_dir / out_file).exists(), f"Output file {out_file} was not created"
    if target_fmt == "jsonl":
        assert jsonl_tools.equal(str(tmp_dir / out_file), str(tmp_dir / "test.jsonl")), f"Output file {out_file} does not match the original file test.jsonl"

@pytest.mark.parametrize("fmt,out_file,in_files", [
    pytest.param("jinx", "test.joined.jinx", ["test.jsonl.jinx", "test.jsonl.mds.jinx"], marks=pytest.mark.dependency(depends=["convert_jsonl_jinx", "convert_mds_jinx"], scope="session")),
    pytest.param("jsonl", "test.joined.jsonl.gz", ["test.jsonl.parquet.jsonl", "test.jsonl.mds.jsonl"], marks=pytest.mark.dependency(depends=["convert_parquet_jsonl", "convert_mds_jsonl"], scope="session")),
    pytest.param("mds", "test.joined.mds", ["test.jsonl.mds", "test.jsonl.parquet.mds"], marks=pytest.mark.dependency(depends=["convert_jsonl_mds", "convert_parquet_mds"], scope="session")),
    pytest.param("msgpack", "test.joined.msgpack", ["test.jsonl.msgpack", "test.jsonl.mds.msgpack"], marks=pytest.mark.dependency(depends=["convert_jsonl_msgpack", "convert_mds_msgpack"], scope="session")),
    pytest.param("parquet", "test.joined.parquet", ["test.jsonl.parquet", "test.jsonl.mds.parquet"], marks=pytest.mark.dependency(depends=["convert_jsonl_parquet", "convert_mds_parquet"], scope="session")),
])
def test_join(fmt, out_file, in_files, tmp_dir):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "join",
        fmt,
        str(tmp_dir / out_file),
        *[str(tmp_dir / f) for f in in_files],
        "--overwrite",
        "--yes"
    ])
    assert result.exit_code == 0, f"Failed joining files for {fmt}: {result.output}"
    assert (tmp_dir / out_file).exists(), f"Output file {out_file} was not created"

@pytest.mark.parametrize("fmt,in_files", [
    pytest.param("jinx", ["test.jsonl.jinx", "test.jsonl.mds.jinx"], marks=pytest.mark.dependency(depends=["convert_jsonl_jinx", "convert_mds_jinx"], scope="session")),
    pytest.param("jsonl", ["test.jsonl", "test.jsonl.mds.jsonl"], marks=pytest.mark.dependency(depends=["convert_mds_jsonl"], scope="session")),
    pytest.param("mds", ["test.jsonl.mds", "test.jsonl.parquet.mds"], marks=pytest.mark.dependency(depends=["convert_jsonl_mds", "convert_parquet_mds"], scope="session")),
    pytest.param("msgpack", ["test.jsonl.msgpack", "test.jsonl.mds.msgpack"], marks=pytest.mark.dependency(depends=["convert_jsonl_msgpack", "convert_mds_msgpack"], scope="session")),
    pytest.param("parquet", ["test.jsonl.parquet", "test.jsonl.mds.parquet"], marks=pytest.mark.dependency(depends=["convert_jsonl_parquet", "convert_mds_parquet"], scope="session")),
])
def test_split(fmt, in_files, tmp_dir):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "split",
        fmt,
        *[str(tmp_dir / f) for f in in_files],
        "--prefix",
        f"test_{fmt}",
        "--output-dir",
        str(tmp_dir),
        "--overwrite",
        "--yes"
    ])
    assert result.exit_code == 0, f"Failed splitting files for {fmt}: {result.output}"
    assert [None for f in tmp_dir.iterdir() if f.name.startswith(f"test_{fmt}")]
