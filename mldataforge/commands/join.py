import click
from datasets import load_dataset

from ..options import *
from ..utils import *

__all__ = ["join"]

@click.group()
def join():
    pass

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("jsonl_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option("infer", ["none", "infer", "pigz", "gzip", "bz2", "xz"])
@processes_option()
@overwrite_option()
@yes_option()
def jsonl(output_file, jsonl_files, compression, processes, overwrite, yes):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_jsonl(
        load_dataset("json", data_files=jsonl_files, split="train"),
        output_file,
        compression=compression,
        processes=processes,
    )

@join.command()
@click.argument("output_dir", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(None, ['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'])
@processes_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@buf_size_option()
@no_bulk_option()
def mds(output_dir, mds_directories, compression, processes, overwrite, yes, batch_size, buf_size, no_bulk):
    check_arguments(output_dir, overwrite, yes, mds_directories)
    save_mds(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk),
        output_dir,
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        pigz=use_pigz(compression),
    )

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option("snappy", ["snappy", "gzip", "zstd"])
@overwrite_option()
@yes_option()
@batch_size_option()
def parquet(output_file, parquet_files, compression, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_parquet(
        load_dataset("parquet", data_files=parquet_files, split="train"),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
