import click

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["msgpack_to_jinx", "msgpack_to_jsonl", "msgpack_to_mds", "msgpack_to_parquet"]

@click.group()
def msgpack():
    pass

@msgpack.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('msgpack_files', nargs=-1, type=click.Path(exists=True))
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
    msgpack_to_jinx(**kwargs)
def msgpack_to_jinx(output_file, msgpack_files, compression, compression_args, overwrite, yes, shard_size, trafo, compress_threshold, compress_ratio, encoding, binary_threshold, ext_sep):
    check_arguments(output_file, overwrite, yes, msgpack_files)
    save_jinx(
        load_msgpack_files(msgpack_files),
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

@msgpack.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("msgpack_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
def jsonl(**kwargs):
    msgpack_to_jsonl(**kwargs)
def msgpack_to_jsonl(output_file, msgpack_files, compression, compression_args, overwrite, yes, trafo):
    check_arguments(output_file, overwrite, yes, msgpack_files)
    save_jsonl(
        load_msgpack_files(msgpack_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        trafo=trafo,
    )

@msgpack.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('msgpack_files', nargs=-1, type=click.Path(exists=True))
@compression_option(MDS_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@shard_size_option()
@no_pigz_option()
@trafo_option()
def mds(**kwargs):
    msgpack_to_mds(**kwargs)
def msgpack_to_mds(output_dir, msgpack_files, compression, compression_args, overwrite, yes, buf_size, shard_size, no_pigz, trafo):
    check_arguments(output_dir, overwrite, yes, msgpack_files)
    save_mds(
        load_msgpack_files(msgpack_files),
        output_dir,
        compression=compression,
        compression_args=compression_args,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz),
        shard_size=shard_size,
        trafo=trafo,
    )

@msgpack.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('msgpack_files', nargs=-1, type=click.Path(exists=True))
@compression_option(PARQUET_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@trafo_option()
def parquet(**kwargs):
    msgpack_to_parquet(**kwargs)
def msgpack_to_parquet(output_file, msgpack_files, compression, compression_args, overwrite, yes, batch_size, trafo):
    check_arguments(output_file, overwrite, yes, msgpack_files)
    save_parquet(
        load_msgpack_files(msgpack_files),
        output_file,
        compression=compression,
        compression_args=compression_args,
        batch_size=batch_size,
        trafo=trafo,
    )
