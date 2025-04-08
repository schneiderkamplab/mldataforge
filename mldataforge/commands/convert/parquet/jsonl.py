#!/usr/bin/env python
import click
from datasets import concatenate_datasets, load_dataset
from mltiming import timing
import os
from pygz import PigzFile
from shutil import which

from ....utils import create_temp_file

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("parquet_files", type=click.Path(exists=True), required=True, nargs=-1)
@click.option("--compression", default="infer", type=click.Choice(["none", "infer", "pigz", "gzip", "bz2", "xz"]), help="Compress the output JSONL file (default: infer; pigz for parallel gzip).")
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing JSONL files.")
def jsonl(output_file, parquet_files, compression, processes, overwrite):
    if os.path.exists(output_file) and not overwrite:
        raise click.BadParameter(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
    if not parquet_files:
        raise click.BadArgumentUsage("No parquet files provided.")
    dss = []
    for parquet_file in parquet_files:
        with timing(message=f"Loading from {parquet_file}"):
            ds = load_dataset("parquet", data_files=parquet_file, split="train")
        dss.append(ds)
    if len(dss) == 1:
        ds = dss[0]
    else:
        with timing(message=f"Concatenating {len(dss)} datasets"):
            ds = concatenate_datasets(dsets=dss)
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
    with timing(message=f"Saving to {output_file} with compression {compression} and {processes} processes"):
        ds.to_json(output_file, num_proc=processes, orient="records", lines=True, compression=compression)
    if orig_output_file is not None:
        with timing(message=f"Compressing {output_file} to {orig_output_file} with pigz using {processes} processes"):
            with open(output_file, "rt") as f_in, PigzFile(f"{orig_output_file}.gz", "wt", threads=processes) as f_out:
                for line in f_in:
                    f_out.write(line)
