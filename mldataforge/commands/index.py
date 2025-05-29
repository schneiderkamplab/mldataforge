import click

from ..indexing import *
from ..options import *
from ..utils import *

__all__ = ["identity", "join", "random", "reverse", "slice"]

@click.group()
def index():
    pass

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), nargs=-1)
@overwrite_option()
@yes_option()
@number_option()
@percentage_option()
@offset_option()
@every_option()
def identity(**kwargs):
    index_identity(**kwargs)
def index_identity(output_file, mds_directories, overwrite, yes, number, percentage, offset, every):
    check_arguments(output_file, overwrite, yes)
    max_index = get_max_index(number, mds_directories)
    indices = identity_permutation(max_index)
    indices = process_indices(indices, every=every, offset=offset, number=number, percentage=percentage)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("input_files", type=click.Path(exists=True), nargs=-1)
@overwrite_option()
@yes_option()
@number_option()
@percentage_option()
@offset_option()
@every_option()
def join(**kwargs):
    index_join(**kwargs)
def index_join(output_file, input_files, overwrite, yes, number, percentage, offset, every):
    check_arguments(output_file, overwrite, yes)
    indices = join_indices(input_files)
    indices = process_indices(indices, every=every, offset=offset, number=number, percentage=percentage)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), nargs=-1)
@overwrite_option()
@yes_option()
@split_option()
@shuffle_option()
@number_option()
@percentage_option()
@offset_option()
@every_option()
def random(**kwargs):
    index_random(**kwargs)
def index_random(output_file, mds_directories, overwrite, yes, split, shuffle, number, percentage, offset, every):
    check_arguments(output_file, overwrite, yes)
    max_index = get_max_index(number, mds_directories, split=split)
    indices = shuffle_permutation(max_index, seed=shuffle)
    indices = process_indices(indices, every=every, offset=offset, number=number, percentage=percentage)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("input_file", type=click.Path(exists=True), required=True)
@overwrite_option()
@yes_option()
@number_option()
@percentage_option()
@offset_option()
@every_option()
def reverse(**kwargs):
    index_reverse(**kwargs)
def index_reverse(output_file, input_file, overwrite, yes, number, percentage, offset, every):
    check_arguments(output_file, overwrite, yes)
    indices = load_index(input_file)
    indices = reverse_permutation(indices)
    indices = process_indices(indices, every=every, offset=offset, number=number, percentage=percentage)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), nargs=-1)
@overwrite_option()
@yes_option()
@split_option()
@batch_size_option()
@reader_option()
@number_option()
@percentage_option()
@offset_option()
@every_option()
@shuffle_option(default=42)
def shuffle(**kwargs):
    index_shuffle(**kwargs)
def index_shuffle(output_file, mds_directories, overwrite, yes, split, batch_size, reader, number, percentage, offset, every, shuffle):
    check_arguments(output_file, overwrite, yes)
    ds = load_mds_directories(
        mds_directories,
        split=split,
        batch_size=batch_size,
        reader=reader,
    )
    indices = shuffle_permutation(len(ds), seed=shuffle)
    indices = process_indices(indices, every=every, offset=offset, number=number, percentage=percentage)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), nargs=-1)
@overwrite_option()
@yes_option()
@split_option()
@batch_size_option()
@reader_option()
@number_option()
@percentage_option()
@offset_option()
@every_option()
@sort_key_option()
def sort(**kwargs):
    index_sort(**kwargs)
def index_sort(output_file, mds_directories, overwrite, yes, split, batch_size, reader, number, percentage, offset, every, sort_key):
    check_arguments(output_file, overwrite, yes)
    ds = load_mds_directories(
        mds_directories,
        split=split,
        batch_size=batch_size,
        reader=reader,
    )
    indices = sort_permutation(ds, sort_key)
    indices = process_indices(indices, every=every, offset=offset, number=number, percentage=percentage)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("input_file", type=click.Path(exists=True), required=True)
@click.option("--remainder", type=click.Path(exists=False), default=None, help="File to save the remainder indices")
@overwrite_option()
@yes_option()
@number_option()
@percentage_option()
@offset_option()
@every_option()
def slice(**kwargs):
    index_slice(**kwargs)
def index_slice(output_file, input_file, overwrite, yes, number, percentage, offset, every, remainder):
    check_arguments(output_file, overwrite, yes)
    all_indices = load_index(input_file)
    indices = process_indices(all_indices, every=every, offset=offset, number=number, percentage=percentage)
    save_index(indices, output_file)
    if remainder is not None:
        remainder_indices = compute_remainder(all_indices, indices)
        save_index(remainder_indices, remainder)
