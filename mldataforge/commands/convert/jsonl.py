import click
from datasets import load_dataset

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["jsonl_to_mds", "jsonl_to_parquet"]

@click.group()
def jsonl():
    pass

@jsonl.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option(MDS_COMPRESSIONS)
@overwrite_option()
@yes_option()
@processes_option()
@buf_size_option()
@shard_size_option()
@no_pigz_option()
def mds(**kwargs):
    jsonl_to_mds(**kwargs)
def jsonl_to_mds(output_dir, jsonl_files, compression, processes, overwrite, yes, buf_size, shard_size, no_pigz):
    check_arguments(output_dir, overwrite, yes, jsonl_files)
    save_mds(
        load_jsonl_files(jsonl_files),
        output_dir,
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz),
        shard_size=shard_size,
    )

@jsonl.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option(PARQUET_COMPRESSIONS)
@overwrite_option()
@yes_option()
@batch_size_option()
def parquet(**kwargs):
    jsonl_to_parquet(**kwargs)
def jsonl_to_parquet(output_file, jsonl_files, compression, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_parquet(
        load_jsonl_files(jsonl_files),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
