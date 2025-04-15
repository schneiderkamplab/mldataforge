from mldataforge.commands.join import join_jsonl, join_mds, join_parquet
import pytest

@pytest.mark.parametrize("fmt,compression,out_file,in_file", [
    ("jsonl", None, "test.jsonl", "test.parquet.jsonl.gz"),
    ("jsonl", "none", "test.jsonl", "test.parquet.jsonl.gz"),
    ("jsonl", "brotli", "test.jsonl.br", "test.parquet.jsonl.gz"),
    ("jsonl", "bz2", "test.jsonl.bz2", "test.parquet.jsonl.gz"),
    ("jsonl", "gzip", "test.jsonl.gz", "test.parquet.jsonl.gz"),
    ("jsonl", "lz4", "test.jsonl.lz4", "test.parquet.jsonl.gz"),
    ("jsonl", "lzma", "test.jsonl.lzma", "test.parquet.jsonl.gz"),
    ("jsonl", "pigz", "test.jsonl.gz", "test.parquet.jsonl.gz"),
    ("jsonl", "snappy", "test.jsonl.snappy", "test.parquet.jsonl.gz"),
    ("jsonl", "xz", "test.jsonl.xz", "test.parquet.jsonl.gz"),
    ("jsonl", "zstd", "test.jsonl.zst", "test.parquet.jsonl.gz"),
    ("mds", None, "test.none.mds", "test.parquet.mds"),
    ("mds", "none", "test.none.mds", "test.parquet.mds"),
    ("mds", "brotli", "test.br.mds", "test.parquet.mds"),
    ("mds", "bz2", "test.bz2.mds", "test.parquet.mds"),
    ("mds", "gzip", "test.gzip.mds", "test.parquet.mds"),
    ("mds", "pigz", "test.pigz.mds", "test.parquet.mds"),
    ("mds", "snappy", "test.snappy.mds", "test.parquet.mds"),
    ("mds", "zstd", "test.zstd.mds", "test.parquet.mds"),
    ("parquet", "brotli", "test.br.parquet", "test.parquet"),
    ("parquet", "gzip", "test.gzip.parquet", "test.parquet"),
    ("parquet", "lz4", "test.lz4.parquet", "test.parquet"),
    ("parquet", "snappy", "test.snappy.parquet", "test.parquet"),
    ("parquet", "zstd", "test.zstd.parquet", "test.parquet"),
])
def test_compression(fmt, compression, out_file, in_file, tmp_dir):
    if fmt == "jsonl":
        join_jsonl(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression=compression,
            processes=64,
            overwrite=True,
            yes=True,
        )
    elif fmt == "mds":
        join_mds(
            output_dir=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression=compression,
            processes=64,
            overwrite=True,
            yes=True,
            batch_size=2**16,
            buf_size=2**20,
            no_bulk=False,
            no_pigz=True,
        )
    elif fmt == "parquet":
        join_parquet(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression=compression,
            overwrite=True,
            yes=True,
            batch_size=2**16,
        )
    assert (tmp_dir / out_file).exists(), f"Output file {out_file} was not created"
    if (tmp_dir / out_file).is_file():
        assert (tmp_dir / out_file).stat().st_size > 2**20, f"Output file {out_file} is too small"
    else:
        assert sum(f.stat().st_size for f in (tmp_dir / out_file).glob("*.mds*")) > 2**20, f"Output directory {out_file} is too small"
