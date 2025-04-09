import click

from ....utils import check_arguments, load_mds_directories, save_jsonl

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@click.option("--compression", default="infer", type=click.Choice(["none", "infer", "pigz", "gzip", "bz2", "xz"]), help="Compress the output JSONL file (default: infer; pigz for parallel gzip).")
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing JSONL files.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove files without confirmation.")
@click.option("--batch-size", default=2**16, help="Batch size for loading MDS directories (default: 65536).")
def jsonl(output_file, mds_directories, compression, processes, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, mds_directories)
    save_jsonl(
        load_mds_directories(mds_directories, batch_size=batch_size),
        output_file,
        compression=compression,
        processes=processes,
    )
