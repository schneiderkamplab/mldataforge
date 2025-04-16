import click
from datasets import load_dataset

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["parquet_to_jsonl", "parquet_to_mds"]

@click.group()
def parquet():
    pass

@parquet.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(JSONL_COMPRESSIONS)
@processes_option()
@overwrite_option()
@yes_option()
def jsonl(**kwargs):
    parquet_to_jsonl(**kwargs)
def parquet_to_jsonl(output_file, parquet_files, compression, processes, overwrite, yes):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_jsonl(
        load_dataset("parquet", data_files=parquet_files, split="train"),
        output_file,
        compression=compression,
        processes=processes,
    )

@parquet.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('parquet_files', nargs=-1, type=click.Path(exists=True))
@compression_option(MDS_COMPRESSIONS)
@processes_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@shard_size_option()
@no_pigz_option()
def mds(**kwargs):
    parquet_to_mds(**kwargs)
def parquet_to_mds(output_dir, parquet_files, compression, processes, overwrite, yes, buf_size, shard_size, no_pigz):
    check_arguments(output_dir, overwrite, yes, parquet_files)
    save_mds(
        load_dataset("parquet", data_files=parquet_files, split="train"),
        output_dir,
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz=no_pigz),
        shard_size=shard_size,
    )
