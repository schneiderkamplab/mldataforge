import click

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["mds_to_jsonl", "mds_to_parquet"]

@click.group()
def mds():
    pass

@mds.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@no_bulk_option()
@trafo_option()
@shuffle_option()
@index_option()
def jsonl(**kwargs):
    mds_to_jsonl(**kwargs)
def mds_to_jsonl(output_file, mds_directories, compression, compression_args, overwrite, yes, batch_size, no_bulk, trafo, shuffle, index):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_jsonl(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk, shuffle=shuffle, index=index),
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
@batch_size_option()
@no_bulk_option()
@trafo_option()
@shuffle_option()
@index_option()
def parquet(**kwargs):
    mds_to_parquet(**kwargs)
def mds_to_parquet(output_file, mds_directories, compression, compression_args, overwrite, yes, batch_size, no_bulk, trafo, shuffle, index):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_parquet(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk, shuffle=shuffle, index=index),
        output_file,
        compression=compression,
        compression_args=compression_args,
        batch_size=batch_size,
        trafo=trafo,
    )
