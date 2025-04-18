import click
from datasets import load_dataset

from ..compression import *
from ..options import *
from ..utils import *

__all__ = ["join_jsonl", "join_mds", "join_parquet"]

@click.group()
def join():
    pass

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("jsonl_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(JSONL_COMPRESSIONS)
@processes_option()
@overwrite_option()
@yes_option()
def jsonl(**kwargs):
    join_jsonl(**kwargs)
def join_jsonl(output_file, jsonl_files, compression, processes, overwrite, yes):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_jsonl(
        load_jsonl_files(jsonl_files),
        output_file,
        compression=compression,
        processes=processes,
    )

@join.command()
@click.argument("output_dir", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(MDS_COMPRESSIONS)
@processes_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@buf_size_option()
@no_bulk_option()
@shard_size_option()
@no_pigz_option()
def mds(**kwargs):
    print(kwargs)
    join_mds(**kwargs)
def join_mds(output_dir, mds_directories, compression, processes, overwrite, yes, batch_size, buf_size, no_bulk, shard_size, no_pigz):
    check_arguments(output_dir, overwrite, yes, mds_directories)
    save_mds(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk),
        output_dir,
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        shard_size=shard_size,
        pigz=use_pigz(compression, no_pigz),
    )

@join.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(PARQUET_COMPRESSIONS)
@overwrite_option()
@yes_option()
@batch_size_option()
def parquet(**kwargs):
    join_parquet(**kwargs)
def join_parquet(output_file, parquet_files, compression, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_parquet(
        load_dataset("parquet", data_files=parquet_files, split="train"),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
