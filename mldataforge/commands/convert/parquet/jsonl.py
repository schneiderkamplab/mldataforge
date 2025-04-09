import click
from mltiming import timing

from ....utils import check_overwrite, create_temp_file, determine_compression, load_parquet_files, pigz_compress

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@click.option("--compression", default="infer", type=click.Choice(["none", "infer", "pigz", "gzip", "bz2", "xz"]), help="Compress the output JSONL file (default: infer; pigz for parallel gzip).")
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing JSONL files.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove files without confirmation.")
@click.option("--buf-size", default=2**24, help=f"Buffer size for pigz compression (default: {2**24}).")
def jsonl(output_file, parquet_files, compression, processes, overwrite, yes, buf_size):
    check_overwrite(output_file, overwrite, yes)
    if not parquet_files:
        raise click.BadArgumentUsage("No parquet files provided.")
    ds = load_parquet_files(parquet_files)
    compression = determine_compression(output_file, compression)
    compressed_file = None
    if compression == "pigz":
        compression, compressed_file, output_file = None, output_file, create_temp_file()
    ds.to_json(output_file, num_proc=processes, orient="records", lines=True, compression=compression)
    if compressed_file is not None:
        pigz_compress(output_file, compressed_file, processes, buf_size, keep=False)
