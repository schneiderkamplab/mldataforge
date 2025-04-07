#!/usr/bin/env python
import atexit
import click
from datasets import load_dataset
from mltiming import timing
import os
from pygz import PigzFile
from shutil import which
import tempfile

def create_temp_file():
    # Create a named temp file, don't delete right away
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp_name = temp.name
    # Close so others can open it again without conflicts (especially on Windows)
    temp.close()

    # Schedule its deletion at exit
    atexit.register(_cleanup_file, temp_name)

    return temp_name

def _cleanup_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass

@click.command()
@click.argument("parquet_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(exists=False), required=False)
@click.option("--compression", default="infer", type=click.Choice(["none", "infer", "pigz", "gzip", "bz2", "xz"]), help="Compress the output JSONL file (default: infer; pigz for parallel gzip).")
@click.option("--threads", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing JSONL files.")
def jsonl(parquet_file, output_file, compression, threads, overwrite):
    if os.path.exists(output_file) and not overwrite:
        raise click.ClickException(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
    with timing(message=f"Loading from {parquet_file}"):
        ds = load_dataset("parquet", data_files=parquet_file)
    orig_output_file = None
    if compression == "none":
        compression = None
    elif compression == "infer":
        if output_file.endswith(".gz") and which("pigz") is not None:
            compression = "pigz"
    if compression == "pigz":
        compression = None
        orig_output_file = output_file
        output_file = create_temp_file()
    with timing(message=f"Saving to {output_file} with compression {compression}"):
        ds["train"].to_json(output_file, orient="records", lines=True, compression=compression)
    if orig_output_file is not None:
        with timing(message=f"Compressing {output_file} to {orig_output_file} with pigz using {threads} threads"):
            with open(output_file, "rt") as f_in, PigzFile(f"{orig_output_file}.gz", "wt", threads=threads) as f_out:
                f_out.write(f_in.read())
