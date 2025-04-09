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
def jsonl(output_file, mds_directories, compression, processes, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_jsonl(
        load_mds_directories(mds_directories, batch_size=batch_size),
        output_file,
        compression=compression,
        processes=processes,
    )

@mds.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option("snappy", ["snappy", "gzip", "zstd"])
@overwrite_option()
@yes_option()
@batch_size_option()
def parquet(output_file, parquet_files, compression, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_parquet(
        load_mds_directories(parquet_files, batch_size=batch_size),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
