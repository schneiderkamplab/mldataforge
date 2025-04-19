import click
from datasets import load_dataset

from ..compression import *
from ..options import *
from ..utils import *

__all__ = ["split_jsonl", "split_mds", "split_parquet"]

@click.group()
def split():
    pass

@split.command()
@click.argument("jsonl_files", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(JSONL_COMPRESSIONS)
@processes_option()
@overwrite_option()
@yes_option()
@trafo_option()
def jsonl(*args, **kwargs):
    split_jsonl(*args, **kwargs)
def split_jsonl(jsonl_files, prefix, output_dir, size_hint, compression, processes, overwrite, yes, trafo):
    save_jsonl(
        load_jsonl_files(jsonl_files),
        output_file=f"{output_dir}/{prefix}{{part:04d}}.jsonl{extension_compression(compression, jsonl_files[0])}",
        compression=compression,
        processes=processes,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
        trafo=trafo,
    )

@split.command()
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(MDS_COMPRESSIONS)
@processes_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@batch_size_option()
@no_bulk_option()
@shard_size_option()
@no_pigz_option()
@trafo_option()
@shuffle_option()
def mds(*args, **kwargs):
    split_mds(*args, **kwargs)
def split_mds(mds_directories, prefix, output_dir, size_hint, compression, processes, overwrite, yes, buf_size, batch_size, no_bulk, shard_size, no_pigz, trafo, shuffle):
    save_mds(
        load_mds_directories(mds_directories, batch_size=batch_size, bulk=not no_bulk, shuffle=shuffle),
        output_dir=f"{output_dir}/{prefix}{{part:04d}}",
        processes=processes,
        compression=compression,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz),
        shard_size=shard_size,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
        trafo=trafo,
    )

@split.command()
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@prefix_option()
@output_dir_option()
@size_hint_option()
@compression_option(PARQUET_COMPRESSIONS)
@overwrite_option()
@yes_option()
@batch_size_option()
@trafo_option()
def parquet(*args, **kwargs):
    split_parquet(*args, **kwargs)
def split_parquet(parquet_files, prefix, output_dir, size_hint, compression, overwrite, yes, batch_size, trafo):
    save_parquet(
        load_dataset("parquet", data_files=parquet_files, split="train"),
        output_file=f"{output_dir}/{prefix}{{part:04d}}.parquet",
        compression=compression,
        batch_size=batch_size,
        size_hint=size_hint,
        overwrite=overwrite,
        yes=yes,
        trafo=trafo,
    )