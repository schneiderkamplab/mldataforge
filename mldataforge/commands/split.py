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
        output_file=f"{output_dir}/{prefix}{{part:04d}}.jsonl{extension(compression, jsonl_files[0])}",
        compression=compression,
        processes=processes,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
    )

@split.command()
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(None, ['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'])
@processes_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@batch_size_option()
@no_bulk_option()
@shard_size_option()
def mds(mds_directories, prefix, output_dir, size_hint, compression, processes, overwrite, yes, buf_size, batch_size, no_bulk, shard_size):
    save_mds(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk),
        output_dir=f"{output_dir}/{prefix}{{part:04d}}",
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        pigz=use_pigz(compression),
        shard_size=shard_size,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
    )
