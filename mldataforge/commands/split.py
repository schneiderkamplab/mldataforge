import click
from datasets import load_dataset

from ..options import *
from ..utils import *

__all__ = ["split"]

@click.group()
def split():
    pass

@split.command()
@click.argument("jsonl_files", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option("infer", ["none", "infer", "pigz", "gzip", "bz2", "xz"])
@processes_option()
@overwrite_option()
@yes_option()
def jsonl(jsonl_files, prefix, output_dir, size_hint, compression, processes, overwrite, yes):
    save_jsonl(
        load_dataset("json", data_files=jsonl_files, split="train"),
        output_file=f"{output_dir}/{prefix}{{part}}.jsonl{extension(compression, jsonl_files[0])}",
        compression=compression,
        processes=processes,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
    )
