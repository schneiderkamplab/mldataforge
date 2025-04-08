#!/usr/bin/env python
import click
import json
from mltiming import timing
import os
from shutil import rmtree, which
from streaming import MDSWriter
from tqdm import tqdm

from ....utils import confirm_overwrite, infer_mds_encoding, open_jsonl

@click.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@click.option('--compression', type=click.Choice(['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'], case_sensitive=False), default=None, help='Compression type for the output dataset (default: None).')
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing MDS directory.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove entire directory trees without confirmation.")
def mds(output_dir, jsonl_files, processes, compression, overwrite, yes):
    if os.path.exists(output_dir):
        if not overwrite:
            raise click.BadParameter(f"Output directory '{output_dir}' already exists. Use --overwrite to overwrite.")
        if not yes:
            confirm_overwrite(f"Output directory '{output_dir}' already exists. Do you want to delete this directory and all its contents?")
        rmtree(output_dir)
    if not jsonl_files:
        raise click.BadArgumentUsage("No jsonl files provided.")
    pigz = compression == "pigz" or (compression == "gzip" and which("pigz") is not None)
    if compression == "none" or pigz:
        compression = None
    if compression == "gzip":
        compression = "gz"
    with open_jsonl(jsonl_files[0]) as f:
        sample = json.loads(f.readline())
    columns = {key: infer_mds_encoding(value) for key, value in sample.items()}
    lines = 0
    with MDSWriter(out=output_dir, columns=columns, compression=compression) as writer:
        for jsonl_file in tqdm(jsonl_files):
            with timing(message=f"Processing {jsonl_file}"):
                with open_jsonl(jsonl_file, compression="infer") as f:
                    for line_num, line in enumerate(f, start=1):
                        try:
                            item = json.loads(line)
                            writer.write(item)
                        except json.JSONDecodeError as e:
                            print(f"Skipping line {line_num} in {jsonl_file} due to JSON error: {e}")
                        lines += 1
    print(f"Wrote {lines} lines from {len(jsonl_files)} files to MDS files in {output_dir}")
    if pigz:
        for file in os.listdir(output_dir):
            if file.endswith(".mds"):
                file_path = os.path.join(output_dir, file)
                with timing(message=f"Compressing {file_path} with pigz using {processes} processes"):
                    os.system(f"pigz -p {processes} {file_path}")
        print(f"Compressed {output_dir} with pigz")
