import click
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ....utils import batch_iterable, check_overwrite, load_mds_directories

@click.command()
@click.argument("output_file", type=click.Path(exists=False), required=True)
@click.argument("mds_directories", type=click.Path(exists=True), required=True, nargs=-1)
@click.option("--compression", default="snappy", type=click.Choice(["snappy", "gzip", "zstd"]), help="Compress the Parquet file (default: snappy).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing Parquet files.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove files without confirmation.")
@click.option("--batch-size", default=2**16, help="Batch size for loading MDS directories and writing Parquet files (default: 65536).")
def parquet(output_file, mds_directories, compression, overwrite, yes, batch_size):
    check_overwrite(output_file, overwrite, yes)
    if not mds_directories:
        raise click.BadArgumentUsage("No MDS files provided.")
    ds = load_mds_directories(mds_directories, batch_size=batch_size)
    writer = None
    for batch in tqdm(batch_iterable(ds, batch_size), desc="Writing to Parquet", unit="batch", total=(len(ds)+batch_size-1) // batch_size):
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression=compression)
        writer.write_table(table)
    writer.close()