from click.testing import CliRunner
from mldataforge.commands import cli
import pytest

@pytest.mark.dependency(name="conversion")
@pytest.mark.parametrize("src_fmt,target_fmt,out_file,in_file", [
    ("jsonl", "mds", "test.jsonl.mds", "test.jsonl"),
    ("jsonl", "parquet", "test.jsonl.parquet", "test.jsonl"),
    ("mds", "jsonl", "test.jsonl.mds.jsonl", "test.jsonl.mds"),
    ("mds", "parquet", "test.jsonl.mds.parquet", "test.jsonl.mds"),
    ("parquet", "jsonl", "test.jsonl.parquet.jsonl", "test.jsonl.parquet"),
    ("parquet", "mds", "test.jsonl.parquet.mds", "test.jsonl.parquet"),
])
def test_conversion(src_fmt, target_fmt, out_file, in_file, tmp_dir):
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

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.parametrize("fmt,out_file,in_files", [
    ("jsonl", "test.joined.jsonl.gz", ["test.jsonl.parquet.jsonl", "test.jsonl.mds.jsonl"]),
    ("mds", "test.joined.mds", ["test.jsonl.mds", "test.jsonl.parquet.mds"]),
    ("parquet", "test.joined.parquet", ["test.jsonl.parquet", "test.jsonl.mds.parquet"]),
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

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.parametrize("fmt,in_files", [
    ("jsonl", ["test.jsonl", "test.jsonl.mds.jsonl"]),
    ("mds", ["test.jsonl.mds", "test.jsonl.parquet.mds"]),
    ("parquet", ["test.jsonl.parquet", "test.jsonl.mds.parquet"]),
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
