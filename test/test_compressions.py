from mldataforge.commands.join import join_jsonl, join_mds, join_parquet
from mldataforge.compression import use_pigz
import pytest

@pytest.mark.parametrize("fmt,compression,out_file,in_file", [
    ("jsonl", "none", "test.none.jsonl.gz", "test.jsonl"),
    ("jsonl", "brotli", "test.br.jsonl.gz", "test.jsonl"),
    ("jsonl", "bz2", "test.bz2.jsonl.gz", "test.jsonl"),
    ("jsonl", "gzip", "test.gzip.jsonl.gz", "test.jsonl"),
    ("jsonl", "lz4", "test.lz4.jsonl.gz", "test.jsonl"),
    ("jsonl", "lzma", "test.lzma.jsonl.gz", "test.jsonl"),
    ("jsonl", "pigz", "test.pigz.jsonl.gz", "test.jsonl"),
    ("jsonl", "snappy", "test.snappy.jsonl.gz", "test.jsonl"),
    ("jsonl", "xz", "test.xz.jsonl.gz", "test.jsonl"),
    ("jsonl", "zip", "test.zip.jsonl.gz", "test.jsonl"),
    ("jsonl", "zstd", "test.zstd.jsonl.gz", "test.jsonl"),
    ("mds", "none", "test.none.mds", "test.mds"),
    ("mds", "brotli", "test.br.mds", "test.mds"),
    ("mds", "bz2", "test.bz2.mds", "test.mds"),
    ("mds", "gzip", "test.gzip.mds", "test.mds"),
    ("mds", "pigz", "test.pigz.mds", "test.mds"),
    ("mds", "snappy", "test.snappy.mds", "test.mds"),
    ("mds", "zstd", "test.zstd.mds", "test.mds"),
    ("parquet", "none", "test.none.parquet", "test.parquet"),
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
            buf_size=2**16,
            pigz=use_pigz(compression),
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
    assert (tmp_dir / out_file).stat().st_size > 2**20, f"Output file {out_file} is too small"
