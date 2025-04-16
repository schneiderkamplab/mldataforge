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
@processes_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@no_bulk_option()
def jsonl(**kwargs):
    mds_to_jsonl(**kwargs)
def mds_to_jsonl(output_file, mds_directories, compression, processes, overwrite, yes, batch_size, no_bulk):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_jsonl(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk),
        output_file,
        compression=compression,
        processes=processes,
    )

@mds.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(PARQUET_COMPRESSIONS)
@overwrite_option()
@yes_option()
@batch_size_option()
@no_bulk_option()
def parquet(**kwargs):
    mds_to_parquet(**kwargs)
def mds_to_parquet(output_file, mds_directories, compression, overwrite, yes, batch_size, no_bulk):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_parquet(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
