import click

from ...options import *
from ...utils import *

__all__ = ["mds"]

@click.group()
def mds():
    pass

@mds.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option("infer", ["none", "infer", "pigz", "gzip", "bz2", "xz"])
@processes_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@no_bulk_option()
def jsonl(output_file, mds_directories, compression, processes, overwrite, yes, batch_size, no_bulk):
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
@compression_option("snappy", ["snappy", "gzip", "zstd"])
@overwrite_option()
@yes_option()
@batch_size_option()
@no_bulk_option()
def parquet(output_file, mds_directories, compression, overwrite, yes, batch_size, no_bulk):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_parquet(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
