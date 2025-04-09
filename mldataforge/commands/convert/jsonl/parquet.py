import click
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ....utils import batch_iterable, check_overwrite, open_jsonl

def _iterate(jsonl_files):
    lines = 0
    for jsonl_file in tqdm(jsonl_files, desc="Processing JSONL files", unit="file"):
        with open_jsonl(jsonl_file, compression="infer") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    item = json.loads(line)
                    yield item
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_num} in {jsonl_file} due to JSON error: {e}")
                lines += 1
    print(f"Wrote {lines} lines from {len(jsonl_files)} files")

@click.command()
@click.argument('output_file', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@click.option("--compression", default="snappy", type=click.Choice(["snappy", "gzip", "zstd"]), help="Compress the Parquet file (default: snappy).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing MDS directory.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove entire directory trees without confirmation.")
@click.option("--batch-size", default=2**16, help="Batch size for loading MDS directories and writing Parquet files (default: 65536).")
def parquet(output_file, jsonl_files, compression, overwrite, yes, batch_size):
    check_overwrite(output_file, overwrite, yes)
    if not jsonl_files:
        raise click.BadArgumentUsage("No JSONL files provided.")
    writer = None
    for batch in batch_iterable(_iterate(jsonl_files), batch_size):
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression=compression)
        writer.write_table(table)
    writer.close()