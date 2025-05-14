import bisect
from collections.abc import Sequence
import PIL
import PIL.JpegImagePlugin
import PIL.PngImagePlugin
import click
from datasets import Dataset, load_dataset
import json
import msgpack
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import os
import shutil
from streaming.base import StreamingDataset
from tqdm import tqdm
import yaml

from .compression import determine_compression, open_compression, pigz_compress
from .indexing import IndexedDatasetView, reverse_permutation, shuffle_permutation, sort_permutation
from .jinx import JinxDatasetReader, JinxDatasetWriter
from .lazy_dict import LazyDict
from .mds import MDS_READERS, MDSBulkDatasetReader, MDSRAMDatasetReader, MDSSampleWriter
from .trafos import get_transformations

__all__ = [
    "CFG",
    "ConcatDataset",
    "check_arguments",
    "confirm_overwrite",
    "count_mds",
    "get_max_index",
    "join_indices",
    "load_index",
    "load_jinx_paths",
    "load_jsonl_files",
    "load_mds_directories",
    "load_msgpack_files",
    "load_parquet_files",
    "load_pipeline_config",
    "save_index",
    "save_jinx",
    "save_jsonl",
    "save_mds",
    "save_msgpack",
    "save_parquet",
]

CFG = {
    "progress": True,
    "echo": False,
}

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = [0]
        for ds in datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, index) - 1
        local_index = index - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_index]

    def __iter__(self):
        for ds in self.datasets:
            yield from ds

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
            os.remove(output_path)
            if CFG["echo"]:
                click.echo(f"Deleted existing file '{output_path}'")
        elif os.path.isdir(output_path):
            if not overwrite:
                raise click.BadParameter(f"Output directory '{output_path}' already exists. Use --overwrite to overwrite.")
            if not yes:
                confirm_overwrite(f"Output directory '{output_path}' already exists. Do you want to delete this directory and all its contents?")
            shutil.rmtree(output_path)
            if CFG["echo"]:
                click.echo(f"Deleted existing directory '{output_path}'")
        else:
            raise click.BadParameter(f"Output path '{output_path}' exists but is neither a file nor a directory.")

def confirm_overwrite(message):
    if not click.confirm("Are you sure you want to proceed?"):
        raise click.Abort()

def count_mds(mds_directories, split='.'):
    counter = 0
    for mds_directory in mds_directories:
        index_path = Path(mds_directory) / split / "index.json"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file '{index_path}' not found.")
        with open(index_path, "rt") as f:
            index = json.load(f)
        for shard in index["shards"]:
            counter += shard["samples"]
    return counter

def get_max_index(number, mds_directories, split='.'):
    if mds_directories:
        return count_mds(mds_directories, split=split)
    if number is not None:
        return number
    raise click.BadParameter("Either mds_directories or number must be provided.")

def _ensure_json_encoding(value):
    """Ensure that the value is JSON serializable."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, bytes):
        return value.decode("latin1")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_ensure_json_encoding(item) for item in value]
    if isinstance(value, dict):
        return {_ensure_json_key(key): _ensure_json_encoding(val) for key, val in value.items()}
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)}")

def _ensure_json_key(key):
    """Ensure that the key is a valid JSON key."""
    if isinstance(key, str):
        return key
    if isinstance(key, bytes):
        return key.decode("latin1")
    return str(key)

def _infer_mds_encoding(value):
    """Determine the MDS encoding for a given value."""
    if isinstance(value, str):
        return 'str'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float64'
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, bytes):
        return 'bytes'
    if isinstance(value, np.ndarray):
        return 'ndarray'
    if isinstance(value, np.int8):
        return 'int8'
    if isinstance(value, np.int16):
        return 'int16'
    if isinstance(value, np.int32):
        return 'int32'
    if isinstance(value, np.int64):
        return 'int64'
    if isinstance(value, np.uint8):
        return 'uint8'
    if isinstance(value, np.uint16):
        return 'uint16'
    if isinstance(value, np.uint32):
        return 'uint32'
    if isinstance(value, np.uint64):
        return 'uint64'
    if isinstance(value, np.float16):
        return 'float16'
    if isinstance(value, np.float32):
        return 'float32'
    if isinstance(value, np.float64):
        return 'float64'
    if isinstance(value, PIL.JpegImagePlugin.JpegImageFile):
        return 'JPEG'   
    if isinstance(value, PIL.PngImagePlugin.PngImageFile):
        return 'PNG'
    if isinstance(value, PIL.Image.Image):
        return 'PIL'
    if isinstance(value, Sequence) and all(isinstance(x, bytearray) for x in value):
        return 'jpeg_array'
    if isinstance(value, list) and all(isinstance(x, (PIL.JpegImagePlugin.JpegImageFile)) for x in value):
        return 'list[jpeg]'
    if isinstance(value, list) and all(isinstance(x, (PIL.PngImagePlugin.PngImageFile)) for x in value):
        return 'list[png]'
    if isinstance(value, list) and all(isinstance(x, (PIL.Image.Image)) for x in value):
        return 'list[pil]'
    if isinstance(value, (list, dict, bool, type(None))):
        return 'json'
    return 'pkl'

def join_indices(input_files):
    loaded = []
    total = 0
    for file in input_files:
        arr = load_index(file)
        loaded.append(arr)
        total += len(arr)
    indices = np.empty(total, dtype=np.uint64)
    offset = 0
    while loaded:
        arr = loaded.pop(0)
        indices[offset:offset+len(arr)] = arr
        offset += len(arr)
    return indices

def _limit_iterable(iterable, limit):
    for i, item in enumerate(iterable):
        if i >= limit:
            break
        yield item

def load_index(input_file):
    with open(input_file, "rb") as f:
        indices = np.load(f)
    return indices

def load_jinx_paths(jinx_paths, split=None, shuffle=None, index=None, sort_key=None, lazy=False, trafo=None, mmap=False, encoding=None):
    if shuffle is not None:
        if index is not None:
            raise click.BadArgumentUsage("Cannot use index and shuffling simultaneously.")
        if sort_key is not None:
            raise click.BadArgumentUsage("Cannot use sort key and shuffling simultaneously.")
    if index is not None:
        if sort_key is not None:
            raise click.BadArgumentUsage("Cannot use sort key and indexing simultaneously.")
    ds = JinxDatasetReader(jinx_paths, split=split, lazy=lazy, mmap=mmap, encoding=encoding)
    if shuffle is not None:
        indices = shuffle_permutation(len(ds), seed=abs(shuffle))
        if shuffle < 0:
            indices = reverse_permutation(indices)
        if CFG["echo"]:
            click.echo(f"Created shuffle indices for {len(ds)} samples")
        ds = IndexedDatasetView(ds, indices)
    if index is not None:
        indices = load_index(index)
        if CFG["echo"]:
            click.echo(f"Loaded index with {len(indices)} indices")
        ds = IndexedDatasetView(ds, indices)
    if sort_key is not None:
        indices = sort_permutation(ds, sort_key)
        if CFG["echo"]:
            click.echo(f"Created sort key with {len(indices)} indices")
        ds = IndexedDatasetView(ds, indices)
    ds = get_transformations(trafo)(ds)
    return ds

def load_jsonl_files(jsonl_files):
    compressions = [determine_compression("jsonl", jsonl_file) for jsonl_file in jsonl_files]
    if "br" in compressions or "snappy" in compressions:
        return _streaming_jsonl(jsonl_files, compressions)
    return load_dataset("json", data_files=jsonl_files, split="train")

def load_mds_directories(mds_directories, split='.', batch_size=2**16, reader="ram", shuffle=None, index=None, sort_key=None):
    if shuffle is not None:
        if reader == "bulk":
            raise click.BadArgumentUsage("Bulk reader does not support shuffling by design.")
        if index is not None:
            raise click.BadArgumentUsage("Cannot use index and shuffling simultaneously.")
        if sort_key is not None:
            raise click.BadArgumentUsage("Cannot use sort key and shuffling simultaneously.")
    if index is not None:
        if reader == "bulk":
            raise click.BadArgumentUsage("Bulk reader does not support indexing by design.")
        if sort_key is not None:
            raise click.BadArgumentUsage("Cannot use sort key and indexing simultaneously.")
    if sort_key is not None:
        if reader == "bulk":
            raise click.BadArgumentUsage("Bulk reader does not support sorting by design.")
    if reader == "bulk":
        return MDSBulkDatasetReader(mds_directories, split=split)
    if reader == "ram":
        ds = MDSRAMDatasetReader(mds_directories, split=split)
    elif reader == "streaming":
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
            ds = ConcatDataset(dss)
            if CFG["echo"]:
                click.echo(f"Concatenated {len(dss)} datasets")
    else:
        raise click.BadArgumentUsage(f"Invalid reader: {reader}. Supported readers are {MDS_READERS['choices']}.")
    if shuffle is not None:
        indices = shuffle_permutation(len(ds), seed=abs(shuffle))
        if shuffle < 0:
            indices = reverse_permutation(indices)
        if CFG["echo"]:
            click.echo(f"Created shuffle indices for {len(ds)} samples")
        ds = IndexedDatasetView(ds, indices)
    if index is not None:
        indices = load_index(index)
        if CFG["echo"]:
            click.echo(f"Loaded index with {len(indices)} indices")
        ds = IndexedDatasetView(ds, indices)
    if sort_key is not None:
        indices = sort_permutation(ds, sort_key)
        if CFG["echo"]:
            click.echo(f"Created sort key with {len(indices)} indices")
        ds = IndexedDatasetView(ds, indices)
    return ds

def load_msgpack_files(msgpack_files):
    compressions = [determine_compression("msgpack", msgpack_file) for msgpack_file in msgpack_files]
    return _streaming_msgpack(msgpack_files, compressions)

def load_parquet_files(parquet_files):
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    return ds

def load_pipeline_config(pipeline_config):
    cfg_path = Path(pipeline_config)
    with open(pipeline_config, "rt") as f:
        if cfg_path.suffix.lower() in (".yaml", ".yml"):
            cfg = yaml.safe_load(f)
        elif cfg_path.suffix.lower() == ".json":
            cfg = json.load(f)
        else:
            raise click.BadParameter(f"Invalid pipeline config file (neither yaml nor json): {pipeline_config}")
    assert isinstance(cfg, dict)
    return cfg

def save_index(indices, output_file, overwrite=True, yes=True):
    with open(output_file, "wb") as f:
        np.save(f, indices)

def save_jinx(iterable, output_file, compression=None, compression_args={"processes": 64}, shard_size=None, size_hint=None, overwrite=True, yes=True, trafo=None, compress_ratio=0.67, compress_threshold=128, encoding="a85", binary_threshold=None, ext_sep="."):
    compression = determine_compression("jinx", output_file, compression)
    writer = None
    part = 0
    trafo = get_transformations(trafo)
    for sample in tqdm(trafo(iterable), desc="Writing to JINX", unit="sample", disable=not CFG["progress"]):
        if writer is None:
            part_file = output_file.format(part=part)
            check_arguments(part_file, overwrite, yes)
            writer = JinxDatasetWriter(part_file, shard_size=shard_size, compression=compression, index_compression=compression, compress_threshold=compress_threshold, compress_ratio=compress_ratio, encoding=encoding, binary_threshold=binary_threshold, ext_sep=ext_sep)
            offset = 0
        prev = writer.tell()
        writer.write(sample)
        post = writer.tell()
        offset += (post - prev) if prev < post else post
        if size_hint is not None and offset >= size_hint:
            writer.close()
            part += 1
            writer = None
    if writer is not None:
        writer.close()

def save_jsonl(iterable, output_file, compression=None, compression_args={"processes": 64}, size_hint=None, overwrite=True, yes=True, trafo=None):
    f = None
    part = 0
    trafo = get_transformations(trafo)
    for item in tqdm(trafo(iterable), desc="Writing to JSONL", unit="sample", disable=not CFG["progress"]):
        if f is None:
            part_file = output_file.format(part=part)
            check_arguments(part_file, overwrite, yes)
            f = open_compression(part_file, mode="wb", compression=compression, compression_args=compression_args)
        if isinstance(item, LazyDict):
            item = item.materialize()
        item = _ensure_json_encoding(item)
        f.write(f"{json.dumps(item)}\n".encode("utf-8"))
        if size_hint is not None and f.tell() >= size_hint:
            f.close()
            part += 1
            f = None
    if f is not None:
        f.close()

def save_mds(it, output_dir, compression=None, compression_args={"processes": 64}, buf_size=2**24, pigz=True, shard_size=None, size_hint=None, overwrite=True, yes=True, trafo=None):
    compression = determine_compression("mds", output_dir, compression, no_pigz=not pigz)
    if shard_size is not None and shard_size > 2**31:
        shard_size = 2**31
    writer = None
    part = 0
    files = []
    trafo = get_transformations(trafo)
    for sample in tqdm(trafo(it), desc="Writing to MDS", unit="sample", disable=not CFG["progress"]):
        if writer is None:
            part_dir = output_dir.format(part=part)
            check_arguments(part_dir, overwrite, yes)
            files.append(part_dir)
            columns = {key: _infer_mds_encoding(value) for key, value in sample.items()}
            writer = MDSSampleWriter(out=part_dir, columns=columns, compression=compression, size_limit=shard_size)
            offset = 0
        prev = writer.new_shard_size
        if isinstance(sample, LazyDict):
            sample = sample.materialize()
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
            for file_name in tqdm(file_names, desc="Compressing with pigz", unit="file", disable=not CFG["progress"]):
                compressed_file_name = file_name + ".gz"
                file_path = os.path.join(output_dir, file_name)
                compressed_file_path = os.path.join(output_dir, compressed_file_name)
                pigz_compress(file_path, compressed_file_path, compression_args.get("processes", 64), buf_size=buf_size, keep=False)
                name2info[file_name]["compression"] = "gz"
                name2info[file_name]["zip_data"] = {
                    "basename": compressed_file_name,
                    "bytes": os.stat(compressed_file_path).st_size,
                    "hashes": {},
                }
            json.dump(index, open(index_path, "wt"))
            if CFG["echo"]:
                click.echo(f"Compressed {index_path} with pigz")

def save_msgpack(iterable, output_file, compression=None, compression_args={"processes": 64}, size_hint=None, overwrite=True, yes=True, trafo=None):
    f = None
    part = 0
    trafo = get_transformations(trafo)
    for item in tqdm(trafo(iterable), desc="Writing to JSONL", unit="sample", disable=not CFG["progress"]):
        if f is None:
            part_file = output_file.format(part=part)
            check_arguments(part_file, overwrite, yes)
            f = open_compression(part_file, mode="wb", compression=compression, compression_args=compression_args)
        if isinstance(item, LazyDict):
            item = item.materialize()
        packed = msgpack.packb(item, use_bin_type=True)
        f.write(packed)
        if size_hint is not None and f.tell() >= size_hint:
            f.close()
            part += 1
            f = None
    if f is not None:
        f.close()

def save_parquet(it, output_file, compression=None, compression_args={"processes": 64}, batch_size=2**16, size_hint=None, overwrite=True, yes=True, trafo=None):
    compression = determine_compression("parquet", output_file, compression)
    writer = None
    part = 0
    trafo = get_transformations(trafo)
    it = tqdm(it, desc="Writing to Parquet", unit="sample", disable=not CFG["progress"])
    for batch in _batch_iterable(trafo(it), batch_size):
        if isinstance(batch[0], LazyDict):
            for i in range(len(batch)):
                batch[i] = batch[i].materialize()
        table = pa.Table.from_pylist(batch)
        if writer is None:
            part_file = output_file.format(part=part)
            check_arguments(part_file, overwrite, yes)
            writer = pq.ParquetWriter(part_file, table.schema, compression=compression, compression_level=compression_args.get("level", None))
            offset = 0
        writer.write_table(table)
        offset += table.nbytes
        if size_hint is not None and offset >= size_hint:
            writer.close()
            part += 1
            writer = None
    if writer is not None:
        writer.close()

def _streaming_jsonl(jsonl_files, compressions):
    for jsonl_file, compression in tqdm(zip(jsonl_files, compressions), desc="Loading JSONL files", unit="file", disable=not CFG["progress"]):
        for line in open_compression(jsonl_file, mode="rt", compression=compression):
            yield json.loads(line)

def _streaming_msgpack(msgpack_files, compressions):
    for msgpack_file, compression in tqdm(zip(msgpack_files, compressions), desc="Loading MessagePack files", unit="file", disable=not CFG["progress"]):
        with open_compression(msgpack_file, mode="rb", compression=compression) as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for item in unpacker:
                yield item
