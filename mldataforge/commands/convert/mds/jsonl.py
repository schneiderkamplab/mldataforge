import click

from ....options import *
from ....utils import *

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@compression_option("infer", ["none", "infer", "pigz", "gzip", "bz2", "xz"])
@processes_option()
@overwrite_option()
@yes_option()
@batch_size_option()
def jsonl(output_file, mds_directories, compression, processes, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_jsonl(
        load_mds_directories(mds_directories, batch_size=batch_size),
        output_file,
        compression=compression,
        processes=processes,
    )
