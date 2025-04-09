import click
import json
import os
from streaming import MDSWriter
from tqdm import tqdm

from ....utils import check_overwrite, infer_mds_encoding, iterate_jsonl, open_jsonl, pigz_compress, use_pigz

@click.command()
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('jsonl_files', nargs=-1, type=click.Path(exists=True))
@click.option('--compression', type=click.Choice(['none', 'br', 'bz2', 'gzip', 'pigz', 'snappy', 'zstd'], case_sensitive=False), default=None, help='Compression type for the output dataset (default: None).')
@click.option("--processes", default=64, help="Number of processes to use for pigz compression (default: 64).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing MDS directory.")
@click.option("--yes", is_flag=True, help="Assume yes to all prompts. Use with caution as it will remove entire directory trees without confirmation.")
@click.option("--buf-size", default=2**24, help=f"Buffer size for pigz compression (default: {2**24}).")
def mds(output_dir, jsonl_files, processes, compression, overwrite, yes, buf_size):
    check_overwrite(output_dir, overwrite, yes)
    if not jsonl_files:
        raise click.BadArgumentUsage("No JSONL files provided.")
    with open_jsonl(jsonl_files[0]) as f:
        sample = json.loads(f.readline())
    pigz = use_pigz(compression)
    if compression == "none" or pigz:
        compression = None
    if compression == "gzip":
        compression = "gz"
    columns = {key: infer_mds_encoding(value) for key, value in sample.items()}
    lines = 0
    with MDSWriter(out=output_dir, columns=columns, compression=compression) as writer:
        for item in iterate_jsonl(jsonl_files):
            writer.write(item)
    if pigz:
        index_path = os.path.join(output_dir, "index.json")
        index = json.load(open(index_path, "rt"))
        name2info = {shard["raw_data"]["basename"]: shard for shard in index["shards"]}
        file_names = [file for file in os.listdir(output_dir) if file.endswith(".mds")]
        assert set(file_names) == set(name2info.keys())
        for file_name in tqdm(file_names, desc="Compressing with pigz", unit="file"):
            compressed_file_name = file_name + ".gz"
            file_path = os.path.join(output_dir, file_name)
            compressed_file_path = os.path.join(output_dir, compressed_file_name)
            pigz_compress(file_path, compressed_file_path, processes, buf_size=buf_size, keep=False, quiet=True)
            name2info[file_name]["compression"] = "gz"
            name2info[file_name]["zip_data"] = {
                "basename": compressed_file_name,
                "bytes": os.stat(compressed_file_path).st_size,
                "hashes": {},
            }
        json.dump(index, open(index_path, "wt"))
        print(f"Compressed {output_dir} with pigz")
