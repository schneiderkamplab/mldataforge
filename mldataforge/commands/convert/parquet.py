import click

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["parquet_to_jinx", "parquet_to_jsonl", "parquet_to_mds", "parquet_to_msgpack"]

@click.group()
def parquet():
    pass

@parquet.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('parquet_files', nargs=-1, type=click.Path(exists=True))
@compression_option(JINX_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@size_hint_option()
@shard_size_option(default=None)
@trafo_option()
@compress_threshold_option()
@compress_ratio_option()
@binary_threshold_option()
def jinx(**kwargs):
    parquet_to_jinx(**kwargs)
def parquet_to_jinx(output_file, parquet_files, compression, compression_args, overwrite, yes, size_hint, shard_size, trafo, compress_threshold, compress_ratio, binary_threshold):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_jinx(
        load_parquet_files(parquet_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        size_hint=size_hint,
        shard_size=shard_size,
        trafo=trafo,
        compress_threshold=compress_threshold,
        compress_ratio=compress_ratio,
        binary_threshold=binary_threshold,
    )

@parquet.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def jsonl(**kwargs):
    parquet_to_jsonl(**kwargs)
def parquet_to_jsonl(output_file, parquet_files, compression, compression_args, overwrite, yes, trafo):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_jsonl(
        load_parquet_files(parquet_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )

@parquet.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('parquet_files', nargs=-1, type=click.Path(exists=True))
@compression_option(MDS_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@shard_size_option()
@no_pigz_option()
@trafo_option()
def mds(**kwargs):
    parquet_to_mds(**kwargs)
def parquet_to_mds(output_dir, parquet_files, compression, compression_args, overwrite, yes, buf_size, shard_size, no_pigz, trafo):
    check_arguments(output_dir, overwrite, yes, parquet_files)
    save_mds(
        load_parquet_files(parquet_files),
        output_dir,
        compression=compression,
        compression_args=compression_args,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz=no_pigz),
        shard_size=shard_size,
        trafo=trafo,
    )

@parquet.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(MSGPACK_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def msgpack(**kwargs):
    parquet_to_msgpack(**kwargs)
def parquet_to_msgpack(output_file, parquet_files, compression, compression_args, overwrite, yes, trafo):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_msgpack(
        load_parquet_files(parquet_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )
