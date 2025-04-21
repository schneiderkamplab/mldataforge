import click
from datasets import concatenate_datasets, load_dataset
import json
from mltiming import timing
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import os
import shutil
from streaming import StreamingDataset
from tqdm import tqdm

from .compression import determine_compression, open_compression, pigz_compress
from .indexing import IndexedDatasetView, reverse_permutation, shuffle_permutation
from .mds import MDSBulkReader, MDSWriter
from .trafos import Transformations

__all__ = [
    "check_arguments",
    "confirm_overwrite",
    "count_mds",
    "load_index",
    "load_jsonl_files",
    "load_mds_directories",
    "load_parquet_files",
    "save_index",
    "save_jsonl",
    "save_mds",
    "save_parquet",
]

_NO_PROGESS = False
def set_progress(value):
    global _NO_PROGESS
    _NO_PROGESS = value

def _batch_iterable(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch.clear()
    if batch:
        yield batch

def check_arguments(output_path, overwrite, yes, input_paths=None):
    if input_paths is not None and not input_paths:
        raise click.BadArgumentUsage("No input paths provided.")
    if os.path.exists(output_path):
        if os.path.isfile(output_path):
            if not overwrite:
                raise click.BadParameter(f"Output file '{output_path}' already exists. Use --overwrite to overwrite.")
            if not yes:
                confirm_overwrite(f"Output file '{output_path}' already exists. Do you want to delete it?")
            with timing(message=f"Deleting existing file '{output_path}'"):
                os.remove(output_path)
        elif os.path.isdir(output_path):
            if not overwrite:
                raise click.BadParameter(f"Output directory '{output_path}' already exists. Use --overwrite to overwrite.")
            if not yes:
                confirm_overwrite(f"Output directory '{output_path}' already exists. Do you want to delete this directory and all its contents?")
            with timing(message=f"Deleting existing directory '{output_path}'"):
                shutil.rmtree(output_path)
        else:
            raise click.BadParameter(f"Output path '{output_path}' exists but is neither a file nor a directory.")

def confirm_overwrite(message):
    print(message)
    response = input("Are you sure you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        raise click.Abort()

def count_mds(mds_directories):
    counter = 0
    for mds_directory in mds_directories:
        index_path = Path(mds_directory) / "index.json"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file '{index_path}' not found.")
        with open(index_path, "rt") as f:
            index = json.load(f)
        for shard in index["shards"]:
            counter += shard["samples"]
    return counter

def _infer_mds_encoding(value):
    """Determine the MDS encoding for a given value."""
    if isinstance(value, str):
        return 'str'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float32'
    if isinstance(value, bool):
        return 'bool'
    return 'pkl'

def _streaming_jsonl(jsonl_files, compressions):
    for jsonl_file, compression in tqdm(zip(jsonl_files, compressions), desc="Loading JSONL files", unit="file", disable=_NO_PROGESS):
        for line in open_compression(jsonl_file, mode="rt", compression=compression):
            yield json.loads(line)

def _limit_iterable(iterable, limit):
    for i, item in enumerate(iterable):
        if i >= limit:
            break
        yield item

def load_index(input_file):
    with open(input_file, "rb") as f:
        indices = np.load(f)
    return indices

def load_jsonl_files(jsonl_files):
    compressions = [determine_compression("jsonl", jsonl_file) for jsonl_file in jsonl_files]
    if "br" in compressions or "snappy" in compressions:
        return _streaming_jsonl(jsonl_files, compressions)
    return load_dataset("json", data_files=jsonl_files, split="train")

def load_mds_directories(mds_directories, split='.', batch_size=2**16, bulk=True, shuffle=None, index=None):
    if shuffle is not None:
        if bulk:
            raise ValueError("Bulk reader does not support shuffling by design.")
        if index is not None:
            raise ValueError("Cannot use index and shuffling simultaneously.")
    if bulk:
        return MDSBulkReader(mds_directories, split=split)
    dss = []
    for mds_directory in mds_directories:
        ds = StreamingDataset(
            local=mds_directory,
            remote=None,
            split=split,
            shuffle=False,
            allow_unsafe_types=True,
            batch_size=batch_size,
            download_retry=1,
            validate_hash=False,
        )
        dss.append(ds)
    if len(dss) == 1:
        ds = dss[0]
    else:
        with timing(message=f"Concatenating {len(dss)} datasets"):
            ds = concatenate_datasets(dsets=dss)
    if shuffle is not None:
        with timing(message="Creating shuffle indices"):
            indices = shuffle_permutation(len(ds), seed=abs(shuffle))
            if shuffle < 0:
                indices = reverse_permutation(indices)
        ds = IndexedDatasetView(ds, indices)
    if index is not None:
        with timing(message="Loading index"):
            indices = load_index(index)
        ds = IndexedDatasetView(ds, indices)
    return ds

def load_parquet_files(parquet_files):
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    return ds

def save_index(indices, output_file, overwrite=True, yes=True):
    with open(output_file, "wb") as f:
        np.save(f, indices)

def save_jsonl(iterable, output_file, compression=None, processes=64, size_hint=None, overwrite=True, yes=True, trafo=None):
    f = None
    part = 0
    trafo = Transformations([] if trafo is None else [trafo])
    for item in tqdm(trafo(iterable), desc="Writing to JSONL", unit="sample", disable=_NO_PROGESS):
        if f is None:
            part_file = output_file.format(part=part)
            check_arguments(part_file, overwrite, yes)
            f = open_compression(part_file, mode="wb", compression=compression, processes=processes)
        item = _sort_nested(item)
        f.write(f"{json.dumps(item)}\n".encode("utf-8"))
        if size_hint is not None and f.tell() >= size_hint:
            f.close()
            part += 1
            f = None
    if f is not None:
        f.close()

def save_mds(it, output_dir, processes=64, compression=None, buf_size=2**24, pigz=True, shard_size=None, size_hint=None, overwrite=True, yes=True, trafo=None):
    compression = determine_compression("mds", output_dir, compression, no_pigz=not pigz)
    if shard_size is not None and shard_size > 2**31:
        shard_size = 2**31
    writer = None
    part = 0
    files = []
    trafo = Transformations([] if trafo is None else [trafo])
    for sample in tqdm(trafo(it), desc="Writing to MDS", unit="sample", disable=_NO_PROGESS):
        if writer is None:
            part_dir = output_dir.format(part=part)
            check_arguments(part_dir, overwrite, yes)
            files.append(part_dir)
            columns = {key: _infer_mds_encoding(value) for key, value in sample.items()}
            writer = MDSWriter(out=part_dir, columns=columns, compression=compression, size_limit=shard_size)
            offset = 0
        prev = writer.new_shard_size
        writer.write(sample)
        offset += (writer.new_shard_size - prev) if prev < writer.new_shard_size else writer.new_shard_size
        if size_hint is not None and offset >= size_hint:
            writer.finish()
            part += 1
            writer = None
    if writer is not None:
        writer.finish()
    if pigz:
        for output_dir in files:
            index_path = os.path.join(output_dir, "index.json")
            index = json.load(open(index_path, "rt"))
            name2info = {shard["raw_data"]["basename"]: shard for shard in index["shards"]}
            file_names = [file for file in os.listdir(output_dir) if file.endswith(".mds")]
            assert set(file_names) == set(name2info.keys())
            for file_name in tqdm(file_names, desc="Compressing with pigz", unit="file", disable=_NO_PROGESS):
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

def save_parquet(it, output_file, compression=None, batch_size=2**16, size_hint=None, overwrite=True, yes=True, trafo=None):
    compression = determine_compression("parquet", output_file, compression)
    writer = None
    part = 0
    trafo = Transformations([] if trafo is None else [trafo])
    it = tqdm(it, desc="Writing to Parquet", unit="sample", disable=_NO_PROGESS)
    for batch in _batch_iterable(trafo(it), batch_size):
        table = pa.Table.from_pylist(batch)
        if writer is None:
            part_file = output_file.format(part=part)
            check_arguments(part_file, overwrite, yes)
            writer = pq.ParquetWriter(part_file, table.schema, compression=compression)
            offset = 0
        writer.write_table(table)
        offset += table.nbytes
        if size_hint is not None and offset >= size_hint:
            writer.close()
            part += 1
            writer = None
    if writer is not None:
        writer.close()

def _sort_nested(obj):
    if isinstance(obj, dict):
        return {k: _sort_nested(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return [_sort_nested(item) for item in obj]
    else:
        return obj
