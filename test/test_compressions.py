import filecmp
from mldataforge.commands.join import join_jsonl, join_mds, join_parquet
from mldataforge.commands.convert.jsonl import jsonl_to_parquet
from mldataforge.commands.convert.mds import mds_to_parquet
from mldataforge.commands.convert.parquet import parquet_to_jsonl
import pytest

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.dependency(name="compression")
@pytest.mark.parametrize("fmt,compression,out_file,in_file", [
    ("jsonl", None, "test.None.jsonl", "test.parquet.jsonl.gz"),
    ("jsonl", "none", "test.none.jsonl", "test.parquet.jsonl.gz"),
    ("jsonl", "brotli", "test.jsonl.br", "test.parquet.jsonl.gz"),
    ("jsonl", "bz2", "test.jsonl.bz2", "test.parquet.jsonl.gz"),
    ("jsonl", "gzip", "test.jsonl.gz", "test.parquet.jsonl.gz"),
    ("jsonl", "lz4", "test.jsonl.lz4", "test.parquet.jsonl.gz"),
    ("jsonl", "lzma", "test.jsonl.lzma", "test.parquet.jsonl.gz"),
    ("jsonl", "pigz", "test.pigz.jsonl.gz", "test.parquet.jsonl.gz"),
    ("jsonl", "snappy", "test.jsonl.snappy", "test.parquet.jsonl.gz"),
    ("jsonl", "xz", "test.jsonl.xz", "test.parquet.jsonl.gz"),
    ("jsonl", "zstd", "test.jsonl.zst", "test.parquet.jsonl.gz"),
    ("mds", None, "test.None.mds", "test.parquet.mds"),
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
    ("mds", "sample::brotli", "test.sample::brotli.mds", "test.parquet.mds"),
    ("mds", "sample::bz2", "test.sample::bz2.mds", "test.parquet.mds"),
    ("mds", "sample::gzip", "test.sample::gzip.mds", "test.parquet.mds"),
    ("mds", "sample::snappy", "test.sample::snappy.mds", "test.parquet.mds"),
    ("mds", "sample::zstd", "test.sample::zstd.mds", "test.parquet.mds"),
])
def test_compression(fmt, compression, out_file, in_file, tmp_dir, scale_factor):
    if fmt == "jsonl":
        join_jsonl(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression=compression,
            processes=64,
            overwrite=True,
            yes=True,
            trafo=None,
        )
    elif fmt == "mds":
        join_mds(
            output_dir=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression=compression,
            processes=64,
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            buf_size=2**14*scale_factor,
            no_bulk=False,
            shard_size=2**14*scale_factor,
            no_pigz=True,
            trafo=None,
            shuffle=None,
        )
    elif fmt == "parquet":
        join_parquet(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression=compression,
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            trafo=None,
        )
    assert (tmp_dir / out_file).exists(), f"Output file {out_file} was not created"
    if (tmp_dir / out_file).is_file():
        assert (tmp_dir / out_file).stat().st_size > 2**14, f"Output file {out_file} is too small"
    else:
        assert sum(f.stat().st_size for f in (tmp_dir / out_file).glob("*.mds*")) > 2**14, f"Output directory {out_file} is too small"

@pytest.mark.dependency(depends=["compression"])
@pytest.mark.parametrize("fmt,out_file,in_file", [
    ("jsonl", "test.jsonl.br.mds", "test.jsonl.br"),
    ("jsonl", "test.jsonl.bz2.parquet", "test.jsonl.bz2"),
    ("jsonl", "test.jsonl.gz.mds", "test.jsonl.gz"),
    ("jsonl", "test.jsonl.lz4.mds", "test.jsonl.lz4"),
    ("jsonl", "test.jsonl.lzma.mds", "test.jsonl.lzma"),
    ("jsonl", "test.jsonl.pigz.mds", "test.pigz.jsonl.gz"),
    ("jsonl", "test.jsonl.snappy.mds", "test.jsonl.snappy"),
    ("jsonl", "test.jsonl.xz.mds", "test.jsonl.xz"),
    ("jsonl", "test.jsonl.zst.mds", "test.jsonl.zst"),
    ("mds", "test.br.mds.parquet", "test.br.mds"),
    ("mds", "test.bz2.mds.parquet", "test.bz2.mds"),
    ("mds", "test.gzip.mds.parquet", "test.gzip.mds"),
    ("mds", "test.pigz.mds.parquet", "test.pigz.mds"),
    ("mds", "test.snappy.mds.parquet", "test.snappy.mds"),
    ("mds", "test.zstd.mds.parquet", "test.zstd.mds"),
    ("parquet", "test.br.parquet.jsonl", "test.br.parquet"),
    ("parquet", "test.gzip.parquet.jsonl", "test.gzip.parquet"),
    ("parquet", "test.lz4.parquet.jsonl", "test.lz4.parquet"),
    ("parquet", "test.snappy.parquet.jsonl", "test.snappy.parquet"),
    ("parquet", "test.zstd.parquet.jsonl", "test.zstd.parquet"),
    ("mds", "test.sample::brotli.mds.parquet", "test.sample::brotli.mds"),
    ("mds", "test.sample::bz2.mds.parquet", "test.sample::bz2.mds"),
    ("mds", "test.sample::gzip.mds.parquet", "test.sample::gzip.mds"),
    ("mds", "test.sample::snappy.mds.parquet", "test.sample::snappy.mds"),
    ("mds", "test.sample::zstd.mds.parquet", "test.sample::zstd.mds"),
])
def test_decompression(fmt, out_file, in_file, tmp_dir, scale_factor):
    if fmt == "jsonl":
        jsonl_to_parquet(
            output_file=str(tmp_dir / out_file),
            jsonl_files=[str(tmp_dir / in_file)],
            compression="snappy",
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            trafo=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.parquet"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to input file {in_file}"
    elif fmt == "mds":
        mds_to_parquet(
            output_file=str(tmp_dir / out_file),
            mds_directories=[str(tmp_dir / in_file)],
            compression="snappy",
            overwrite=True,
            yes=True,
            batch_size=2**10*scale_factor,
            no_bulk=True,
            trafo=None,
            shuffle=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.parquet"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to input file {in_file}"
    elif fmt == "parquet":
        parquet_to_jsonl(
            output_file=str(tmp_dir / out_file),
            parquet_files=[str(tmp_dir / in_file)],
            compression=None,
            processes=64,
            overwrite=True,
            yes=True,
            trafo=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.none.jsonl"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to input file {in_file}"
    assert (tmp_dir / out_file).exists(), f"Output file {out_file} was not created"
    if (tmp_dir / out_file).is_file():
        assert (tmp_dir / out_file).stat().st_size > 2**14, f"Output file {out_file} is too small"
    else:
        assert sum(f.stat().st_size for f in (tmp_dir / out_file).glob("*.mds*")) > 2**14, f"Output directory {out_file} is too small"
