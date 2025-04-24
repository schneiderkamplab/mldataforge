import filecmp
from mldataforge.commands.join import join_jsonl, join_mds, join_parquet
from mldataforge.commands.convert.mds import mds_to_jsonl
from mldataforge.commands.convert.parquet import parquet_to_jsonl
import pytest

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.dependency(name="compression")
@pytest.mark.parametrize("fmt,compression,out_file,in_file", [
    ("jsonl", None, "test.None.jsonl", "test.jsonl"),
    ("jsonl", "none", "test.none.jsonl", "test.jsonl"),
    ("jsonl", "brotli", "test.jsonl.br", "test.jsonl"),
    ("jsonl", "bz2", "test.jsonl.bz2", "test.jsonl"),
    ("jsonl", "gzip", "test.jsonl.gz", "test.jsonl"),
    ("jsonl", "infer", "test.infer.jsonl.gz", "test.jsonl"),
    ("jsonl", "lz4", "test.jsonl.lz4", "test.jsonl"),
    ("jsonl", "lzma", "test.jsonl.lzma", "test.jsonl"),
    ("jsonl", "pigz", "test.pigz.jsonl.gz", "test.jsonl"),
    ("jsonl", "snappy", "test.jsonl.snappy", "test.jsonl"),
    ("jsonl", "xz", "test.jsonl.xz", "test.jsonl"),
    ("jsonl", "zstd", "test.jsonl.zst", "test.jsonl"),
    ("mds", None, "test.None.mds", "test.jsonl.mds"),
    ("mds", "none", "test.none.mds", "test.jsonl.mds"),
    ("mds", "brotli", "test.br.mds", "test.jsonl.mds"),
    ("mds", "bz2", "test.bz2.mds", "test.jsonl.mds"),
    ("mds", "gzip", "test.gzip.mds", "test.jsonl.mds"),
    ("mds", "pigz", "test.pigz.mds", "test.jsonl.mds"),
    ("mds", "snappy", "test.snappy.mds", "test.jsonl.mds"),
    ("mds", "zstd", "test.zstd.mds", "test.jsonl.mds"),
    ("parquet", "brotli", "test.br.parquet", "test.jsonl.parquet"),
    ("parquet", "gzip", "test.gzip.parquet", "test.jsonl.parquet"),
    ("parquet", "lz4", "test.lz4.parquet", "test.jsonl.parquet"),
    ("parquet", "snappy", "test.snappy.parquet", "test.jsonl.parquet"),
    ("parquet", "zstd", "test.zstd.parquet", "test.jsonl.parquet"),
    ("mds", "sample::brotli", "test.sample.brotli.mds", "test.jsonl.mds"),
    ("mds", "sample::bz2", "test.sample.bz2.mds", "test.jsonl.mds"),
    ("mds", "sample::gzip", "test.sample.gzip.mds", "test.jsonl.mds"),
    ("mds", "sample::snappy", "test.sample.snappy.mds", "test.jsonl.mds"),
    ("mds", "sample::zstd", "test.sample.zstd.mds", "test.jsonl.mds"),
])
def test_compression(fmt, compression, out_file, in_file, tmp_dir, scale_factor, compressions):
    if compressions is not None and compression not in compressions:
        pytest.skip(f"Compression {compression} is not in the list of compressions to test: {compressions}")
    if fmt == "jsonl":
        join_jsonl(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression=compression,
            compression_args={"processes": 64, "quality": 1, "compression_level": 1},
            overwrite=True,
            yes=True,
            trafo=None,
        )
    elif fmt == "mds":
        join_mds(
            output_dir=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression=compression,
            compression_args={"processes": 64, "quality": 1, "compression_level": 1},
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            buf_size=2**14*scale_factor,
            no_bulk=False,
            shard_size=2**14*scale_factor,
            no_pigz=True,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=None,
        )
    elif fmt == "parquet":
        join_parquet(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression=compression,
            compression_args={"processes": 64, "quality": 1, "compression_level": 1},
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            trafo=None,
        )
    assert (tmp_dir / out_file).exists(), f"Output file {out_file} was not created"

@pytest.mark.dependency(depends=["compression"])
@pytest.mark.parametrize("fmt,out_file,in_file", [
    ("jsonl", "test.jsonl.br.jsonl", "test.jsonl.br"),
    ("jsonl", "test.jsonl.bz2.jsonl", "test.jsonl.bz2"),
    ("jsonl", "test.jsonl.gz.jsonl", "test.jsonl.gz"),
    ("jsonl", "test.infer.jsonl.gz.jsonl", "test.infer.jsonl.gz"),
    ("jsonl", "test.jsonl.lz4.jsonl", "test.jsonl.lz4"),
    ("jsonl", "test.jsonl.lzma.jsonl", "test.jsonl.lzma"),
    ("jsonl", "test.pigz.jsonl.gz.jsonl", "test.pigz.jsonl.gz"),
    ("jsonl", "test.jsonl.snappy.jsonl", "test.jsonl.snappy"),
    ("jsonl", "test.jsonl.xz.jsonl", "test.jsonl.xz"),
    ("jsonl", "test.jsonl.zst.jsonl", "test.jsonl.zst"),
    ("mds", "test.br.mds.jsonl", "test.br.mds"),
    ("mds", "test.bz2.mds.jsonl", "test.bz2.mds"),
    ("mds", "test.gzip.mds.jsonl", "test.gzip.mds"),
    ("mds", "test.pigz.mds.jsonl", "test.pigz.mds"),
    ("mds", "test.snappy.mds.jsonl", "test.snappy.mds"),
    ("mds", "test.zstd.mds.jsonl", "test.zstd.mds"),
    ("parquet", "test.br.parquet.jsonl", "test.br.parquet"),
    ("parquet", "test.gzip.parquet.jsonl", "test.gzip.parquet"),
    ("parquet", "test.lz4.parquet.jsonl", "test.lz4.parquet"),
    ("parquet", "test.snappy.parquet.jsonl", "test.snappy.parquet"),
    ("parquet", "test.zstd.parquet.jsonl", "test.zstd.parquet"),
    ("mds", "test.sample.brotli.mds.jsonl", "test.sample.brotli.mds"),
    ("mds", "test.sample.bz2.mds.jsonl", "test.sample.bz2.mds"),
    ("mds", "test.sample.gzip.mds.jsonl", "test.sample.gzip.mds"),
    ("mds", "test.sample.snappy.mds.jsonl", "test.sample.snappy.mds"),
    ("mds", "test.sample.zstd.mds.jsonl", "test.sample.zstd.mds"),
])
def test_decompression(fmt, out_file, in_file, tmp_dir, scale_factor):
    if fmt == "jsonl":
        join_jsonl(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            trafo=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.jsonl"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to test.jsonl"
    elif fmt == "mds":
        mds_to_jsonl(
            output_file=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            split=".",
            batch_size=2**10*scale_factor,
            no_bulk=True,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.jsonl"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to test.jsonl"
    elif fmt == "parquet":
        parquet_to_jsonl(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            trafo=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.jsonl"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to test.jsonl"
