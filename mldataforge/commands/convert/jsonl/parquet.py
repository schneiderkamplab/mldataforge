import click
from datasets import load_dataset

from ....options import *
from ....utils import *

@click.command()
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
