import click
from tqdm import tqdm

from ....utils import check_arguments, infer_mds_encoding, load_parquet_files, save_mds, use_pigz

@click.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('parquet_files', nargs=-1, type=click.Path(exists=True))
@click.option('--compression', type=click.Choice(['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'], case_sensitive=False), default=None, help='Compression type for the output dataset (default: None).')
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing MDS directory.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove entire directory trees without confirmation.")
@click.option("--buf-size", default=2**24, help=f"Buffer size for pigz compression (default: {2**24}).")
def mds(output_dir, parquet_files, processes, compression, overwrite, yes, buf_size):
    check_arguments(output_dir, overwrite, yes, parquet_files)
    ds = load_parquet_files(parquet_files)
    pigz = use_pigz(compression)
    sample = ds[0]
    columns = {key: infer_mds_encoding(value) for key, value in sample.items()}
    it = tqdm(ds, desc="Writing to MDS", unit="sample")
    save_mds(it, output_dir, columns=columns, processes=processes, compression=compression, buf_size=buf_size, pigz=pigz)
