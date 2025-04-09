import click
from datasets import load_dataset

from ....options import *
from ....utils import *

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option("infer", ["none", "infer", "pigz", "gzip", "bz2", "xz"])
@processes_option()
@overwrite_option()
@yes_option()
def jsonl(output_file, parquet_files, compression, processes, overwrite, yes):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_jsonl(
        load_dataset("parquet", data_files=parquet_files, split="train"),
        output_file,
        compression=compression,
        processes=processes,
    )
