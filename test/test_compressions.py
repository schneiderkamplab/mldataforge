import filecmp
from mldataforge.commands.join import join_jinx, join_jsonl, join_mds, join_msgpack, join_parquet
from mldataforge.commands.convert.jinx import jinx_to_jsonl
from mldataforge.commands.convert.mds import mds_to_jsonl
from mldataforge.commands.convert.msgpack import msgpack_to_jsonl
from mldataforge.commands.convert.parquet import parquet_to_jsonl
import pytest

@pytest.mark.parametrize("fmt,compression,out_file,in_file", [
    pytest.param("jinx", None, "test.None.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "none", "test.none.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "brotli", "test.br.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_br", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "bz2", "test.bz2.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_bz2", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "gzip", "test.gzip.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_gzip", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "lz4", "test.lz4.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_lz4", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "lzma", "test.lzma.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_lzma", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "snappy", "test.snappy.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_snappy", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "xz", "test.xz.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_xz", depends=["convert_jsonl_jinx"], scope="session")),
    pytest.param("jinx", "zstd", "test.zst.jinx", "test.jsonl.jinx", marks=pytest.mark.dependency(name="compress_jinx_zstd", depends=["convert_jsonl_jinx"], scope="session")),
    ("jsonl", None, "test.None.jsonl", "test.jsonl"),
    ("jsonl", "none", "test.none.jsonl", "test.jsonl"),
    pytest.param("jsonl", "brotli", "test.jsonl.br", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_br", scope="session")),
    pytest.param("jsonl", "bz2", "test.jsonl.bz2", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_bz2", scope="session")),
    pytest.param("jsonl", "gzip", "test.jsonl.gz", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_gzip", scope="session")),
    pytest.param("jsonl", "infer", "test.infer.jsonl.gz", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_infer", scope="session")),
    pytest.param("jsonl", "lz4", "test.jsonl.lz4", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_lz4", scope="session")),
    pytest.param("jsonl", "lzma", "test.jsonl.lzma", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_lzma", scope="session")),
    pytest.param("jsonl", "pigz", "test.pigz.jsonl.gz", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_pigz", scope="session")),
    pytest.param("jsonl", "snappy", "test.jsonl.snappy", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_snappy", scope="session")),
    pytest.param("jsonl", "xz", "test.jsonl.xz", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_xz", scope="session")),
    pytest.param("jsonl", "zstd", "test.jsonl.zst", "test.jsonl", marks=pytest.mark.dependency(name="compress_jsonl_zstd", scope="session")),
    pytest.param("mds", None, "test.None.mds", "test.jsonl.mds", marks=pytest.mark.dependency(depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "none", "test.none.mds", "test.jsonl.mds", marks=pytest.mark.dependency(depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "brotli", "test.br.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_br", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "bz2", "test.bz2.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_bz2", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "gzip", "test.gzip.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_gzip", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "pigz", "test.pigz.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_pigz", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "snappy", "test.snappy.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_snappy", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "zstd", "test.zstd.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_zstd", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("msgpack", None, "test.None.msgpack", "test.jsonl.msgpack", marks=pytest.mark.dependency(depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "none", "test.none.msgpack", "test.jsonl.msgpack", marks=pytest.mark.dependency(depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "brotli", "test.msgpack.br", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_br", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "bz2", "test.msgpack.bz2", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_bz2", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "gzip", "test.msgpack.gz", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_gzip", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "infer", "test.infer.msgpack.gz", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_infer", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "lz4", "test.msgpack.lz4", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_lz4", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "lzma", "test.msgpack.lzma", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_lzma", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "pigz", "test.pigz.msgpack.gz", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_pigz", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "snappy", "test.msgpack.snappy", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_snappy", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "xz", "test.msgpack.xz", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_xz", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("msgpack", "zstd", "test.msgpack.zst", "test.jsonl.msgpack", marks=pytest.mark.dependency(name="compress_msgpack_zstd", depends=["convert_jsonl_msgpack"], scope="session")),
    pytest.param("parquet", "brotli", "test.br.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(name="compress_parquet_br", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", "gzip", "test.gzip.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(name="compress_parquet_gzip", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", "lz4", "test.lz4.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(name="compress_parquet_lz4", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", "snappy", "test.snappy.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(name="compress_parquet_snappy", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("parquet", "zstd", "test.zstd.parquet", "test.jsonl.parquet", marks=pytest.mark.dependency(name="compress_parquet_zstd", depends=["convert_jsonl_parquet"], scope="session")),
    pytest.param("mds", "sample::brotli", "test.sample.brotli.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_sample_br", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "sample::bz2", "test.sample.bz2.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_sample_bz2", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "sample::gzip", "test.sample.gzip.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_sample_gzip", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "sample::snappy", "test.sample.snappy.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_sample_snappy", depends=["convert_jsonl_mds"], scope="session")),
    pytest.param("mds", "sample::zstd", "test.sample.zstd.mds", "test.jsonl.mds", marks=pytest.mark.dependency(name="compress_mds_sample_zstd", depends=["convert_jsonl_mds"], scope="session")),
])
def test_compression(fmt, compression, out_file, in_file, tmp_dir, scale_factor, compressions):
    if compressions is not None and compression not in compressions:
        pytest.skip(f"Compression {compression} is not in the list of compressions to test: {compressions}")
    if fmt == "jinx":
        join_jinx(
            output_file=str(tmp_dir / out_file),
            jinx_paths=[str(tmp_dir / in_file)],
            compression=compression,
            compression_args={"processes": 64, "quality": 1, "compression_level": 1},
            overwrite=True,
            yes=True,
            shard_size=None,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=None,
            compress_threshold=2**6,
            compress_ratio=1.0,
            binary_threshold=2**8,
        )
    elif fmt == "jsonl":
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
            reader="ram",
            shard_size=2**14*scale_factor,
            no_pigz=True,
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=None,
        )
    elif fmt == "msgpack":
        join_msgpack(
            output_file=str(tmp_dir / out_file),
            msgpack_files=[str(tmp_dir / in_file)],
            compression=compression,
            compression_args={"processes": 64, "quality": 1, "compression_level": 1},
            overwrite=True,
            yes=True,
            trafo=None,
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

@pytest.mark.parametrize("fmt,out_file,in_file", [
    pytest.param("jinx", "test.br.jinx.jsonl", "test.br.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_br"], scope="session")),
    pytest.param("jinx", "test.bz2.jinx.jsonl", "test.bz2.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_bz2"], scope="session")),
    pytest.param("jinx", "test.gzip.jinx.jsonl", "test.gzip.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_gzip"], scope="session")),
    pytest.param("jinx", "test.jinx.lz4.jsonl", "test.lz4.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_lz4"], scope="session")),
    pytest.param("jinx", "test.jinx.lzma.jsonl", "test.lzma.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_lzma"], scope="session")),
    pytest.param("jinx", "test.jinx.snappy.jsonl", "test.snappy.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_snappy"], scope="session")),
    pytest.param("jinx", "test.jinx.xz.jsonl", "test.xz.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_xz"], scope="session")),
    pytest.param("jinx", "test.jinx.zst.jsonl", "test.zst.jinx", marks=pytest.mark.dependency(depends=["compress_jinx_zstd"], scope="session")),
    pytest.param("jsonl", "test.jsonl.br.jsonl", "test.jsonl.br", marks=pytest.mark.dependency(depends=["compress_jsonl_br"], scope="session")),
    pytest.param("jsonl", "test.jsonl.bz2.jsonl", "test.jsonl.bz2", marks=pytest.mark.dependency(depends=["compress_jsonl_bz2"], scope="session")),
    pytest.param("jsonl", "test.jsonl.gz.jsonl", "test.jsonl.gz", marks=pytest.mark.dependency(depends=["compress_jsonl_gzip"], scope="session")),
    pytest.param("jsonl", "test.infer.jsonl.gz.jsonl", "test.infer.jsonl.gz", marks=pytest.mark.dependency(depends=["compress_jsonl_infer"], scope="session")),
    pytest.param("jsonl", "test.jsonl.lzma.jsonl", "test.jsonl.lzma", marks=pytest.mark.dependency(depends=["compress_jsonl_lzma"], scope="session")),
    pytest.param("jsonl", "test.pigz.jsonl.gz.jsonl", "test.pigz.jsonl.gz", marks=pytest.mark.dependency(depends=["compress_jsonl_pigz"], scope="session")),
    pytest.param("jsonl", "test.jsonl.snappy.jsonl", "test.jsonl.snappy", marks=pytest.mark.dependency(depends=["compress_jsonl_snappy"], scope="session")),
    pytest.param("jsonl", "test.jsonl.xz.jsonl", "test.jsonl.xz", marks=pytest.mark.dependency(depends=["compress_jsonl_xz"], scope="session")),
    pytest.param("jsonl", "test.jsonl.zst.jsonl", "test.jsonl.zst", marks=pytest.mark.dependency(depends=["compress_jsonl_zstd"], scope="session")),
    pytest.param("mds", "test.br.mds.jsonl", "test.br.mds", marks=pytest.mark.dependency(depends=["compress_mds_br"], scope="session")),
    pytest.param("mds", "test.bz2.mds.jsonl", "test.bz2.mds", marks=pytest.mark.dependency(depends=["compress_mds_bz2"], scope="session")),
    pytest.param("mds", "test.gzip.mds.jsonl", "test.gzip.mds", marks=pytest.mark.dependency(depends=["compress_mds_gzip"], scope="session")),
    pytest.param("mds", "test.pigz.mds.jsonl", "test.pigz.mds", marks=pytest.mark.dependency(depends=["compress_mds_pigz"], scope="session")),
    pytest.param("mds", "test.snappy.mds.jsonl", "test.snappy.mds", marks=pytest.mark.dependency(depends=["compress_mds_snappy"], scope="session")),
    pytest.param("mds", "test.zstd.mds.jsonl", "test.zstd.mds", marks=pytest.mark.dependency(depends=["compress_mds_zstd"], scope="session")),
    pytest.param("msgpack", "test.msgpack.br.jsonl", "test.msgpack.br", marks=pytest.mark.dependency(depends=["compress_msgpack_br"], scope="session")),
    pytest.param("msgpack", "test.msgpack.bz2.jsonl", "test.msgpack.bz2", marks=pytest.mark.dependency(depends=["compress_msgpack_bz2"], scope="session")),
    pytest.param("msgpack", "test.msgpack.gz.jsonl", "test.msgpack.gz", marks=pytest.mark.dependency(depends=["compress_msgpack_gzip"], scope="session")),
    pytest.param("msgpack", "test.infer.msgpack.gz.jsonl", "test.infer.msgpack.gz", marks=pytest.mark.dependency(depends=["compress_msgpack_infer"], scope="session")),
    pytest.param("msgpack", "test.msgpack.lz4.jsonl", "test.msgpack.lz4", marks=pytest.mark.dependency(depends=["compress_msgpack_lz4"], scope="session")),
    pytest.param("msgpack", "test.msgpack.lzma.jsonl", "test.msgpack.lzma", marks=pytest.mark.dependency(depends=["compress_msgpack_lzma"], scope="session")),
    pytest.param("msgpack", "test.pigz.msgpack.gz.jsonl", "test.pigz.msgpack.gz", marks=pytest.mark.dependency(depends=["compress_msgpack_pigz"], scope="session")),
    pytest.param("msgpack", "test.msgpack.snappy.jsonl", "test.msgpack.snappy", marks=pytest.mark.dependency(depends=["compress_msgpack_snappy"], scope="session")),
    pytest.param("msgpack", "test.msgpack.xz.jsonl", "test.msgpack.xz", marks=pytest.mark.dependency(depends=["compress_msgpack_xz"], scope="session")),
    pytest.param("msgpack", "test.msgpack.zst.jsonl", "test.msgpack.zst", marks=pytest.mark.dependency(depends=["compress_msgpack_zstd"], scope="session")),
    pytest.param("parquet", "test.br.parquet.jsonl", "test.br.parquet", marks=pytest.mark.dependency(depends=["compress_parquet_br"], scope="session")),
    pytest.param("parquet", "test.gzip.parquet.jsonl", "test.gzip.parquet", marks=pytest.mark.dependency(depends=["compress_parquet_gzip"], scope="session")),
    pytest.param("parquet", "test.lz4.parquet.jsonl", "test.lz4.parquet", marks=pytest.mark.dependency(depends=["compress_parquet_lz4"], scope="session")),
    pytest.param("parquet", "test.snappy.parquet.jsonl", "test.snappy.parquet", marks=pytest.mark.dependency(depends=["compress_parquet_snappy"], scope="session")),
    pytest.param("parquet", "test.zstd.parquet.jsonl", "test.zstd.parquet", marks=pytest.mark.dependency(depends=["compress_parquet_zstd"], scope="session")),
    pytest.param("mds", "test.sample.brotli.mds.jsonl", "test.sample.brotli.mds", marks=pytest.mark.dependency(depends=["compress_mds_sample_br"], scope="session")),
    pytest.param("mds", "test.sample.bz2.mds.jsonl", "test.sample.bz2.mds", marks=pytest.mark.dependency(depends=["compress_mds_sample_bz2"], scope="session")),
    pytest.param("mds", "test.sample.gzip.mds.jsonl", "test.sample.gzip.mds", marks=pytest.mark.dependency(depends=["compress_mds_sample_gzip"], scope="session")),
    pytest.param("mds", "test.sample.snappy.mds.jsonl", "test.sample.snappy.mds", marks=pytest.mark.dependency(depends=["compress_mds_sample_snappy"], scope="session")),
    pytest.param("mds", "test.sample.zstd.mds.jsonl", "test.sample.zstd.mds", marks=pytest.mark.dependency(depends=["compress_mds_sample_zstd"], scope="session")),
])
def test_decompression(fmt, out_file, in_file, tmp_dir, scale_factor):
    if fmt == "jinx":
        jinx_to_jsonl(
            output_file=str(tmp_dir / out_file),
            jinx_paths=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            trafo=None,
            split=None,
            shuffle=None,
            index=None,
            sort_key=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.jsonl"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to test.jsonl"
    elif fmt == "jsonl":
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
            reader="ram",
            trafo=None,
            shuffle=None,
            index=None,
            sort_key=None,
        )
        assert filecmp.cmp(str(tmp_dir / "test.jsonl"), str(tmp_dir / out_file), shallow=False), f"Output file {out_file} is not equal to test.jsonl"
    elif fmt == "msgpack":
        msgpack_to_jsonl(
            output_file=str(tmp_dir / out_file),
            msgpack_files=[str(tmp_dir / in_file)],
            compression=None,
            compression_args={"processes": 64},
            overwrite=True,
            yes=True,
            trafo=None,
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
