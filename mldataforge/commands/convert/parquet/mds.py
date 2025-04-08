#!/usr/bin/env python
import click
import json
import os
from streaming import MDSWriter
from tqdm import tqdm

from ....utils import check_overwrite, infer_mds_encoding, load_parquet_files, pigz_compress, use_pigz

@click.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('parquet_files', nargs=-1, type=click.Path(exists=True))
@click.option('--compression', type=click.Choice(['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'], case_sensitive=False), default=None, help='Compression type for the output dataset (default: None).')
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing MDS directory.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove entire directory trees without confirmation.")
@click.option("--buf-size", default=2**24, help=f"Buffer size for pigz compression (default: {2**24}).")
def mds(output_dir, parquet_files, processes, compression, overwrite, yes, buf_size):
    check_overwrite(output_dir, overwrite, yes)
    if not parquet_files:
        raise click.BadArgumentUsage("No parquet files provided.")
    ds = load_parquet_files(parquet_files)
    pigz = use_pigz(compression)
    sample = ds[0]
    if compression == "none" or pigz:
        compression = None
    if compression == "gzip":
        compression = "gz"
    columns = {key: infer_mds_encoding(value) for key, value in sample.items()}
    lines = 0
    with MDSWriter(out=output_dir, columns=columns, compression=compression) as writer:
        for item in tqdm(ds, desc="Processing samples", unit="sample"):
            writer.write(item)
            lines += 1
    print(f"Wrote {lines} lines from {len(parquet_files)} files to MDS files in {output_dir}")
    if pigz:
        file_paths = []
        for file in os.listdir(output_dir):
            if file.endswith(".mds"):
                file_paths.append(os.path.join(output_dir, file))
        for file_path in tqdm(file_paths, desc="Compressing with pigz", unit="file"):
            pigz_compress(file_path, file_path + ".gz", processes, buf_size=buf_size, keep=False, quiet=True)
        output_dir
        print(f"Compressed {output_dir} with pigz")
