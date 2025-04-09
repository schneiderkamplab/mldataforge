import click
from datasets import load_dataset

from ...options import *
from ...utils import *

__all__ = ["jsonl"]

@click.group()
def jsonl():
    pass

@jsonl.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option(None, ['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'])
@overwrite_option()
@yes_option()
@processes_option()
@buf_size_option()
def mds(output_dir, jsonl_files, compression, processes, overwrite, yes, buf_size):
    check_arguments(output_dir, overwrite, yes, jsonl_files)
    save_mds(
        load_dataset("json", data_files=jsonl_files, split="train"),
        output_dir,
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        pigz=use_pigz(compression),
    )

@jsonl.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option("snappy", ["snappy", "gzip", "zstd"])
@overwrite_option()
@yes_option()
@batch_size_option()
def parquet(output_file, jsonl_files, compression, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    save_parquet(
        load_dataset("json", data_files=jsonl_files, split="train"),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
