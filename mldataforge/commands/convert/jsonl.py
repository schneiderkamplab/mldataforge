import click

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["jsonl_to_jinx", "jsonl_to_mds", "jsonl_to_msgpack", "jsonl_to_parquet"]

@click.group()
def jsonl():
    pass

@jsonl.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option(JINX_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@shard_size_option(default=None)
@trafo_option()
@compress_threshold_option()
@compress_ratio_option()
@encoding_option()
@binary_threshold_option()
@ext_sep_option()
def jinx(**kwargs):
    jsonl_to_jinx(**kwargs)
def jsonl_to_jinx(output_file, jsonl_files, compression, compression_args, overwrite, yes, shard_size, trafo, compress_threshold, compress_ratio, encoding, binary_threshold, ext_sep):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_jinx(
        load_jsonl_files(jsonl_files),
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

@jsonl.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option(MDS_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@shard_size_option()
@no_pigz_option()
@trafo_option()
def mds(**kwargs):
    jsonl_to_mds(**kwargs)
def jsonl_to_mds(output_dir, jsonl_files, compression, compression_args, overwrite, yes, buf_size, shard_size, no_pigz, trafo):
    check_arguments(output_dir, overwrite, yes, jsonl_files)
    save_mds(
        load_jsonl_files(jsonl_files),
        output_dir,
        compression=compression,
        compression_args=compression_args,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz),
        shard_size=shard_size,
        trafo=trafo,
    )

@jsonl.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("jsonl_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(MSGPACK_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def msgpack(**kwargs):
    jsonl_to_msgpack(**kwargs)
def jsonl_to_msgpack(output_file, jsonl_files, compression, compression_args, overwrite, yes, trafo):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_msgpack(
        load_jsonl_files(jsonl_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )

@jsonl.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option(PARQUET_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@trafo_option()
def parquet(**kwargs):
    jsonl_to_parquet(**kwargs)
def jsonl_to_parquet(output_file, jsonl_files, compression, compression_args, overwrite, yes, batch_size, trafo):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_parquet(
        load_jsonl_files(jsonl_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        batch_size=batch_size,
        trafo=trafo,
    )
