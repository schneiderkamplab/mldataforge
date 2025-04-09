import click
from datasets import load_dataset

from ....utils import check_arguments, save_jsonl

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@click.option("--compression", default="infer", type=click.Choice(["none", "infer", "pigz", "gzip", "bz2", "xz"]), help="Compress the output JSONL file (default: infer; pigz for parallel gzip).")
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing JSONL files.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove files without confirmation.")
@click.option("--buf-size", default=2**24, help=f"Buffer size for pigz compression (default: {2**24}).")
def jsonl(output_file, parquet_files, compression, processes, overwrite, yes, buf_size):
    check_arguments(output_file, overwrite, yes, parquet_files)
    save_jsonl(
        load_dataset("parquet", data_files=parquet_files, split="train"),
        output_file,
        compression=compression,
        processes=processes,
    )
