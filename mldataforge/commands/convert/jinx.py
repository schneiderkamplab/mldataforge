import click

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["jsonl_to_mds", "jsonl_to_parquet"]

@click.group()
def jinx():
    pass

@jinx.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jinx_paths', nargs=-1, type=click.Path(exists=True))
@compression_option(JSONL_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
@mmap_option()
@split_option()
@shuffle_option()
@index_option()
@sort_key_option()
@lazy_option()
@override_encoding_option()
def jsonl(**kwargs):
    jinx_to_jsonl(**kwargs)
def jinx_to_jsonl(output_file, jinx_paths, compression, compression_args, overwrite, yes, trafo, mmap, split, shuffle, index, sort_key, lazy, override_encoding):
    check_arguments(output_file, overwrite, yes, jinx_paths)
    save_jsonl(
        load_jinx_paths(jinx_paths, split=split, shuffle=shuffle, index=index, sort_key=sort_key, lazy=lazy, trafo=trafo, mmap=mmap, encoding=override_encoding),
        output_file,
        compression=compression,
        compression_args=compression_args,
    )

@jinx.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('jinx_paths', nargs=-1, type=click.Path(exists=True))
@compression_option(MDS_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@buf_size_option()
@shard_size_option()
@no_pigz_option()
@trafo_option()
@mmap_option()
@shuffle_option()
@index_option()
@sort_key_option()
@lazy_option()
@override_encoding_option()
def mds(**kwargs):
    jinx_to_mds(**kwargs)
def jinx_to_mds(output_dir, jinx_paths, compression, compression_args, overwrite, yes, buf_size, shard_size, no_pigz, trafo, mmap, shuffle, index, sort_key, lazy, override_encoding):
    check_arguments(output_dir, overwrite, yes, jinx_paths)
    save_mds(
        load_jinx_paths(jinx_paths, shuffle=shuffle, index=index, sort_key=sort_key, lazy=lazy, trafo=trafo, mmap=mmap, encoding=override_encoding),
        output_dir,
        compression=compression,
        compression_args=compression_args,
        buf_size=buf_size,
        pigz=use_pigz(compression, no_pigz),
        shard_size=shard_size,
    )

@jinx.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("jinx_paths", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option(MSGPACK_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@trafo_option()
@mmap_option()
@shuffle_option()
@index_option()
@sort_key_option()
@lazy_option()
@override_encoding_option()
def msgpack(**kwargs):
    jinx_to_msgpack(**kwargs)
def jinx_to_msgpack(output_file, jinx_paths, compression, compression_args, overwrite, yes, trafo, mmap, shuffle, index, sort_key, lazy, override_encoding):
    check_arguments(output_file, overwrite, yes, jinx_paths)
    save_msgpack(
        load_jinx_paths(jinx_paths, shuffle=shuffle, index=index, sort_key=sort_key, lazy=lazy, trafo=trafo, mmap=mmap, encoding=override_encoding),
        output_file,
        compression=compression,
        compression_args=compression_args,
    )

@jinx.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jinx_paths', nargs=-1, type=click.Path(exists=True))
@compression_option(PARQUET_COMPRESSIONS)
@compression_args_option()
@overwrite_option()
@yes_option()
@batch_size_option()
@trafo_option()
@mmap_option()
@shuffle_option()
@index_option()
@sort_key_option()
@lazy_option()
@override_encoding_option()
def parquet(**kwargs):
    jinx_to_parquet(**kwargs)
def jinx_to_parquet(output_file, jinx_paths, compression, compression_args, overwrite, yes, batch_size, trafo, mmap, shuffle, index, sort_key, lazy, override_encoding):
    check_arguments(output_file, overwrite, yes, jinx_paths)
    save_parquet(
        load_jinx_paths(jinx_paths, shuffle=shuffle, index=index, sort_key=sort_key, lazy=lazy, trafo=trafo, mmap=mmap, encoding=override_encoding),
        output_file,
        compression=compression,
        compression_args=compression_args,
        batch_size=batch_size,
    )
