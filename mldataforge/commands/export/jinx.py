import click

from ...compression import *
from ...options import *
from ...utils import *

__all__ = ["jsonl_to_mds"]

@click.group()
def jinx():
    pass

@jinx.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jinx_paths', nargs=-1, type=click.Path(exists=True))
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
def pyarrow(**kwargs):
    jinx_to_pyarrow(**kwargs)
def jinx_to_pyarrow(output_file, jinx_paths, overwrite, yes, trafo, mmap, split, shuffle, index, sort_key, lazy, override_encoding):
    check_arguments(output_file, overwrite, yes, jinx_paths)
    export_pyarrow(
        load_jinx_paths(jinx_paths, split=split, shuffle=shuffle, index=index, sort_key=sort_key, lazy=lazy, trafo=trafo, mmap=mmap, encoding=override_encoding),
        output_file,
    )
