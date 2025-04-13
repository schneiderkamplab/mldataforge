from click.testing import CliRunner
from mldataforge.commands import cli
import pytest

@pytest.mark.dependency(name="conversion")
@pytest.mark.parametrize("src_fmt,target_fmt,out_file,in_file", [
    ("parquet", "jsonl", "test.parquet.jsonl.gz", "test.parquet"),
    ("jsonl", "parquet", "test.parquet.jsonl.parquet", "test.parquet.jsonl.gz"),
    ("parquet", "mds", "test.parquet.mds", "test.parquet"),
    ("jsonl", "mds", "test.parquet.jsonl.mds", "test.parquet.jsonl.gz"),
    ("mds", "parquet", "test.parquet.mds.parquet", "test.parquet.mds"),
    ("mds", "jsonl", "test.parquet.mds.jsonl.gz", "test.parquet.mds"),
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
    ("jsonl", "test.joined.jsonl.gz", ["test.parquet.jsonl.gz", "test.parquet.mds.jsonl.gz"]),
    ("mds", "test.joined.mds", ["test.parquet.mds", "test.parquet.jsonl.mds"]),
    ("parquet", "test.joined.parquet", ["test.parquet", "test.parquet.mds.parquet"]),
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
    ("jsonl", ["test.parquet.jsonl.gz", "test.parquet.mds.jsonl.gz"]),
    ("mds", ["test.parquet.mds", "test.parquet.jsonl.mds"]),
    ("parquet", ["test.parquet", "test.parquet.mds.parquet"]),
])
def test_split(fmt, in_files, tmp_dir):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "split",
        fmt,
        *[str(tmp_dir / f) for f in in_files],
        "--prefix",
        "test_",
        "--output-dir",
        str(tmp_dir),
        "--overwrite",
        "--yes"
    ])
    assert result.exit_code == 0, f"Failed splitting files for {fmt}: {result.output}"
    assert [None for f in tmp_dir.iterdir() if f.name.startswith("test_") and f.name.endswith(".jsonl.gz")]
