import click

from ....utils import check_arguments, load_mds_directories, save_parquet

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@click.option("--compression", default="snappy", type=click.Choice(["snappy", "gzip", "zstd"]), help="Compress the Parquet file (default: snappy).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing Parquet files.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove files without confirmation.")
@click.option("--batch-size", default=2**16, help="Batch size for loading MDS directories and writing Parquet files (default: 65536).")
def parquet(output_file, parquet_files, compression, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_parquet(
        load_mds_directories(parquet_files, batch_size=batch_size),
        output_file,
        compression=compression,
        batch_size=batch_size,
    )
