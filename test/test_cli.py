from click.testing import CliRunner
from huggingface_hub import hf_hub_download
from mldataforge.commands import cli
from pathlib import Path
import pytest

@pytest.fixture
def tmp_dir():
    tmp_dir = Path(__file__).resolve().parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    if not (tmp_dir / "test.parquet").exists():
        hf_hub_download(
            repo_type="dataset",
            repo_id="jlpang888/tulu_300k",
            revision="main",
            filename="data/train-00000-of-00001.parquet",
            local_dir=tmp_dir,
        )
        (tmp_dir / "data/train-00000-of-00001.parquet").rename(
            tmp_dir / "test.parquet"
        )
    return tmp_dir

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
