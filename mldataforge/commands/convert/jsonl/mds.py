import click
from datasets import load_dataset

from ....options import *
from ....utils import *

@click.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@compression_option(None, ['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'])
@overwrite_option()
@yes_option()
@processes_option()
@buf_size_option()
def mds(output_dir, jsonl_files, processes, compression, overwrite, yes, buf_size):
    check_arguments(output_dir, overwrite, yes, jsonl_files)
    save_mds(
        load_dataset("json", data_files=jsonl_files, split="train"),
        output_dir,
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        pigz=use_pigz(compression),
    )
