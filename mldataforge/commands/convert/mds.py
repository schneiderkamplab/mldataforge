import click

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["mds_to_jinx", "mds_to_jsonl", "mds_to_msgpack", "mds_to_parquet"]

@click.group()
def mds():
    pass

@mds.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('mds_directories', nargs=-1, type=click.Path(exists=True))
@compression_option(JINX_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@split_option()
@batch_size_option()
@reader_option()
@shard_size_option(default=None)
@trafo_option()
@shuffle_option()
@index_option()
@sort_key_option()
@compress_threshold_option()
@compress_ratio_option()
@encoding_option()
@binary_threshold_option()
@ext_sep_option()
def jinx(**kwargs):
    mds_to_jinx(**kwargs)
def mds_to_jinx(output_file, mds_directories, compression, compression_args, overwrite, yes, split, batch_size, reader, shard_size, trafo, shuffle, index, sort_key, compress_threshold, compress_ratio, encoding, binary_threshold, ext_sep):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_jinx(
        load_mds_directories(mds_directories, split=split, batch_size=batch_size, reader=reader, shuffle=shuffle, index=index, sort_key=sort_key),
        output_file,
        compression=compression,
        compression_args=compression_args,
        shard_size=shard_size,
        trafo=trafo,
        compress_threshold=compress_threshold,
        compress_ratio=compress_ratio,
        encoding=encoding,
        binary_threshold=binary_threshold,
        ext_sep=ext_sep,
    )

@mds.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@split_option()
@batch_size_option()
@reader_option()
@trafo_option()
@shuffle_option()
@index_option()
@sort_key_option()
def jsonl(**kwargs):
    mds_to_jsonl(**kwargs)
def mds_to_jsonl(output_file, mds_directories, compression, compression_args, overwrite, yes, split, batch_size, reader, trafo, shuffle, index, sort_key):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_jsonl(
        load_mds_directories(mds_directories, split=split, batch_size=batch_size, reader=reader, shuffle=shuffle, index=index, sort_key=sort_key),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )

@mds.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(MSGPACK_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@split_option()
@batch_size_option()
@reader_option()
@trafo_option()
@shuffle_option()
@index_option()
@sort_key_option()
def msgpack(**kwargs):
    mds_to_msgpack(**kwargs)
def mds_to_msgpack(output_file, mds_directories, compression, compression_args, overwrite, yes, split, batch_size, reader, trafo, shuffle, index, sort_key):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_msgpack(
        load_mds_directories(mds_directories, split=split, batch_size=batch_size, reader=reader, shuffle=shuffle, index=index, sort_key=sort_key),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )


@mds.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(PARQUET_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@split_option()
@batch_size_option()
@reader_option()
@trafo_option()
@shuffle_option()
@index_option()
@sort_key_option()
def parquet(**kwargs):
    mds_to_parquet(**kwargs)
def mds_to_parquet(output_file, mds_directories, compression, compression_args, overwrite, yes, split, batch_size, reader, trafo, shuffle, index, sort_key):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_parquet(
        load_mds_directories(mds_directories, split=split, batch_size=batch_size, reader=reader, shuffle=shuffle, index=index, sort_key=sort_key),
        output_file,
        compression=compression,
        compression_args=compression_args,
        batch_size=batch_size,
        trafo=trafo,
    )
