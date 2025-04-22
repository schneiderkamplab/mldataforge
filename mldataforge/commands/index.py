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
@offset_option()
@every_option()
def identity(**kwargs):
    index_identity(**kwargs)
def index_identity(output_file, mds_directories, overwrite, yes, number, offset, every):
    check_arguments(output_file, overwrite, yes)
    if mds_directories:
        max_index = count_mds(mds_directories)
    elif number is not None:
        max_index = number
    else:
        raise click.BadParameter("Either mds_directories or number must be provided.")
    indices = identity_permutation(max_index)
    indices = process_indices(indices, every=every, offset=offset, number=number)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("input_files", type=click.Path(exists=True), nargs=-1)
@overwrite_option()
@yes_option()
@number_option()
@offset_option()
@every_option()
def join(**kwargs):
    index_join(**kwargs)
def index_join(output_file, input_files, overwrite, yes, number, offset, every):
    check_arguments(output_file, overwrite, yes)
    indices = join_indices(input_files)
    indices = process_indices(indices, every=every, offset=offset, number=number)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), nargs=-1)
@overwrite_option()
@yes_option()
@shuffle_option()
@number_option()
@offset_option()
@every_option()
def random(**kwargs):
    index_random(**kwargs)
def index_random(output_file, mds_directories, overwrite, yes, shuffle, number, offset, every):
    check_arguments(output_file, overwrite, yes)
    if mds_directories:
        max_index = count_mds(mds_directories)
    elif number is not None:
        max_index = number
    else:
        raise click.BadParameter("Either mds_directories or number must be provided.")
    indices = shuffle_permutation(max_index, seed=shuffle)
    indices = process_indices(indices, every=every, offset=offset, number=number)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("input_file", type=click.Path(exists=True), required=True)
@overwrite_option()
@yes_option()
@number_option()
@offset_option()
@every_option()
def reverse(**kwargs):
    index_reverse(**kwargs)
def index_reverse(output_file, input_file, overwrite, yes, number, offset, every):
    check_arguments(output_file, overwrite, yes)
    indices = load_index(input_file)
    indices = reverse_permutation(indices)
    indices = process_indices(indices, every=every, offset=offset, number=number)
    save_index(indices, output_file)

@index.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("input_file", type=click.Path(exists=True), required=True)
@overwrite_option()
@yes_option()
@number_option()
@offset_option()
@every_option()
def slice(**kwargs):
    index_slice(**kwargs)
def index_slice(output_file, input_file, overwrite, yes, number, offset, every):
    check_arguments(output_file, overwrite, yes)
    indices = load_index(input_file)
    indices = process_indices(indices, every=every, offset=offset, number=number)
    save_index(indices, output_file)
