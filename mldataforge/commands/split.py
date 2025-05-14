import click

from ..compression import *
from ..options import *
from ..utils import *

__all__ = ["split_jsonl", "split_mds", "split_parquet"]

@click.group()
def split():
    pass

@split.command()
@click.argument("jinx_paths", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(JINX_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@shard_size_option(default=None)
@trafo_option()
@mmap_option()
@shuffle_option()
@index_option()
@sort_key_option()
@lazy_option()
@compress_threshold_option()
@compress_ratio_option()
@encoding_option()
@binary_threshold_option()
@ext_sep_option()
@override_encoding_option()
def jinx(*args, **kwargs):
    split_jinx(*args, **kwargs)
def split_jinx(jinx_paths, prefix, output_dir, size_hint, compression, compression_args, overwrite, yes, shard_size, trafo, mmap, shuffle, index, sort_key, lazy, compress_threshold, compress_ratio, encoding, binary_threshold, ext_sep, override_encoding):
    save_jinx(
        load_jinx_paths(jinx_paths, shuffle=shuffle, index=index, sort_key=sort_key, lazy=lazy, trafo=trafo, mmap=mmap, encoding=override_encoding),
        output_file=f"{output_dir}/{prefix}{{part:04d}}.jinx",
        compression=compression,
        compression_args=compression_args,
        size_hint=size_hint,
        shard_size=shard_size,
        overwrite=overwrite,
        yes=yes,
        compress_threshold=compress_threshold,
        compress_ratio=compress_ratio,
        encoding=encoding,
        binary_threshold=binary_threshold,
        ext_sep=ext_sep,
    )

@split.command()
@click.argument("jsonl_files", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def jsonl(*args, **kwargs):
    split_jsonl(*args, **kwargs)
def split_jsonl(jsonl_files, prefix, output_dir, size_hint, compression, compression_args, overwrite, yes, trafo):
    save_jsonl(
        load_jsonl_files(jsonl_files),
        output_file=f"{output_dir}/{prefix}{{part:04d}}.jsonl{extension_compression(compression, jsonl_files[0])}",
        compression=compression,
        compression_args=compression_args,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
        trafo=trafo,
    )

@split.command()
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(MDS_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@batch_size_option()
@reader_option()
@shard_size_option()
@no_pigz_option()
@trafo_option()
@shuffle_option()
@index_option()
@sort_key_option()
def mds(*args, **kwargs):
    split_mds(*args, **kwargs)
def split_mds(mds_directories, prefix, output_dir, size_hint, compression, compression_args, overwrite, yes, buf_size, batch_size, reader, shard_size, no_pigz, trafo, shuffle, index, sort_key):
    save_mds(
        load_mds_directories(mds_directories, batch_size=batch_size, reader=reader, shuffle=shuffle, index=index, sort_key=sort_key),
        output_dir=f"{output_dir}/{prefix}{{part:04d}}",
        compression=compression,
        compression_args=compression_args,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz),
        shard_size=shard_size,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
        trafo=trafo,
    )

@split.command()
@click.argument("msgpack_files", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def msgpack(*args, **kwargs):
    split_msgpack(*args, **kwargs)
def split_msgpack(msgpack_files, prefix, output_dir, size_hint, compression, compression_args, overwrite, yes, trafo):
    save_jsonl(
        load_msgpack_files(msgpack_files),
        output_file=f"{output_dir}/{prefix}{{part:04d}}.jsonl{extension_compression(compression, msgpack_files[0])}",
        compression=compression,
        compression_args=compression_args,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
        trafo=trafo,
    )

@split.command()
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(PARQUET_COMPRESSIONS)
@overwrite_option()
@yes_option()
@batch_size_option()
@trafo_option()
def parquet(*args, **kwargs):
    split_parquet(*args, **kwargs)
def split_parquet(parquet_files, prefix, output_dir, size_hint, compression, overwrite, yes, batch_size, trafo):
    save_parquet(
        load_parquet_files(parquet_files),
        output_file=f"{output_dir}/{prefix}{{part:04d}}.parquet",
        compression=compression,
        batch_size=batch_size,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
        trafo=trafo,
    )