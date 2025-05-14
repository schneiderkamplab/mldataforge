import click

from ..compression import *
from ..options import *
from ..utils import *

__all__ = ["join_jsonl", "join_mds", "join_parquet"]

@click.group()
def join():
    pass

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("jinx_paths", type=click.Path(exists=True), required=True, nargs=-1)
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
def jinx(**kwargs):
    join_jinx(**kwargs)
def join_jinx(output_file, jinx_paths, compression, compression_args, overwrite, yes, shard_size, trafo, mmap, shuffle, index, sort_key, lazy, compress_threshold, compress_ratio, encoding, binary_threshold, ext_sep, override_encoding):
    check_arguments(output_file, overwrite, yes, jinx_paths)
    save_jinx(
        load_jinx_paths(jinx_paths, shuffle=shuffle, index=index, sort_key=sort_key, lazy=lazy, trafo=trafo, mmap=mmap, encoding=override_encoding),
        output_file,
        compression=compression,
        compression_args=compression_args,
        shard_size=shard_size,
        compress_threshold=compress_threshold,
        compress_ratio=compress_ratio,
        encoding=encoding,
        binary_threshold=binary_threshold,
        ext_sep=ext_sep,
    )

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("jsonl_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def jsonl(**kwargs):
    join_jsonl(**kwargs)
def join_jsonl(output_file, jsonl_files, compression, compression_args, overwrite, yes, trafo):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_jsonl(
        load_jsonl_files(jsonl_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )

@join.command()
@click.argument("output_dir", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(MDS_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@buf_size_option()
@reader_option()
@shard_size_option()
@no_pigz_option()
@trafo_option()
@shuffle_option()
@index_option()
@sort_key_option()
def mds(**kwargs):
    join_mds(**kwargs)
def join_mds(output_dir, mds_directories, compression, compression_args, overwrite, yes, batch_size, buf_size, reader, shard_size, no_pigz, trafo, shuffle, index, sort_key):
    check_arguments(output_dir, overwrite, yes, mds_directories)
    save_mds(
        load_mds_directories(mds_directories, batch_size=batch_size, reader=reader, shuffle=shuffle, index=index, sort_key=sort_key),
        output_dir,
        compression=compression,
        compression_args=compression_args,
        buf_size=buf_size,
        shard_size=shard_size,
        pigz=use_pigz(compression, no_pigz),
        trafo=trafo,
    )

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("msgpack_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(MSGPACK_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def msgpack(**kwargs):
    join_msgpack(**kwargs)
def join_msgpack(output_file, msgpack_files, compression, compression_args, overwrite, yes, trafo):
    check_arguments(output_file, overwrite, yes, msgpack_files)
    save_msgpack(
        load_msgpack_files(msgpack_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(PARQUET_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@trafo_option()
def parquet(**kwargs):
    join_parquet(**kwargs)
def join_parquet(output_file, parquet_files, compression, compression_args, overwrite, yes, batch_size, trafo):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_parquet(
        load_parquet_files(parquet_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        batch_size=batch_size,
        trafo=trafo,
    )
