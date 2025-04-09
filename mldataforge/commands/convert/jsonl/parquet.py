import click
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ....utils import batch_iterable, check_arguments, iterate_jsonl

@click.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@click.option("--compression", default="snappy", type=click.Choice(["snappy", "gzip", "zstd"]), help="Compress the Parquet file (default: snappy).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing MDS directory.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove entire directory trees without confirmation.")
@click.option("--batch-size", default=2**16, help="Batch size for loading MDS directories and writing Parquet files (default: 65536).")
def parquet(output_file, jsonl_files, compression, overwrite, yes, batch_size):
    check_arguments(output_file, overwrite, yes, jsonl_files)
    writer = None
    for batch in batch_iterable(iterate_jsonl(jsonl_files), batch_size):
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression=compression)
        writer.write_table(table)
    writer.close()