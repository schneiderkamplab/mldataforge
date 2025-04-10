import bz2
import click
from datasets import concatenate_datasets
import gzip
import json
import lzma
from mltiming import timing
import pyarrow as pa
import pyarrow.parquet as pq
import os
import shutil
from streaming import MDSWriter, StreamingDataset
from tqdm import tqdm

from .mds import MDSBulkReader
from .pigz import pigz_open

__all__ = [
    "batch_iterable",
    "check_arguments",
    "confirm_overwrite",
    "extension",
    "load_mds_directories",
    "save_jsonl",
    "save_mds",
    "save_parquet",
    "use_pigz",
]

def batch_iterable(iterable, batch_size):
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

def _determine_compression(file_path, compression="infer"):
    if compression == "infer":
        compression = _infer_compression(file_path)
    if compression == "none":
        compression = None
    return compression

def extension(compression, file_path):
    """Get the file extension for the given compression type."""
    if compression == "infer":
        compression = _infer_compression(file_path)
    if compression in ("gzip", "pigz"):
        return ".gz"
    if compression == "bz2":
        return ".bz2"
    if compression == "xz":
        return ".xz"
    if compression == "zip":
        return ".zip"
    if compression == "zstd":
        return ".zst"
    if compression is None:
        return ""
    raise ValueError(f"Unsupported compression type: {compression}")

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

def _infer_compression(file_path):
    """Infer the compression type from the file extension."""
    extension = os.path.splitext(file_path)[1]
    if extension.endswith('.gz'):
        if _pigz_available():
            return 'pigz'
        return 'gzip'
    if extension.endswith('.bz2'):
        return 'bz2'
    if extension.endswith('.xz'):
        return 'xz'
    if extension.endswith('.zip'):
        return 'zip'
    if extension.endswith('.zst'):
        return 'zstd'
    return None

def load_mds_directories(mds_directories, split='.', batch_size=2**16, bulk=True):
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
    return ds

def _open_jsonl(file_path, mode="rt", compression="infer", processes=64):
    """Open a JSONL file, handling gzip compression if necessary."""
    compression = _determine_compression(file_path, compression)
    if compression == "gzip":
        return gzip.open(file_path, mode)
    if compression == "pigz":
        return pigz_open(file_path, mode, processes=processes) if mode[0] == "w" else gzip.open(file_path, mode)
    if compression == "bz2":
        return bz2.open(file_path, mode)
    if compression == "xz":
        return lzma.open(file_path, mode)
    if compression is None:
        return open(file_path, mode)
    raise ValueError(f"Unsupported compression type: {compression}")

def _pigz_available():
    """Check if pigz is available on the system."""
    return shutil.which("pigz") is not None

def _pigz_compress(input_file, output_file, processes=64, buf_size=2**24, keep=False, quiet=False):
    """Compress a file using pigz."""
    size = os.stat(input_file).st_size
    num_blocks = (size+buf_size-1) // buf_size
    with open(input_file, "rb") as f_in, pigz_open(output_file, "wb", processes=processes) as f_out:
        for _ in tqdm(range(num_blocks), desc="Compressing with pigz", unit="block", disable=quiet):
            buf = f_in.read(buf_size)
            assert buf
            f_out.write(buf)
        buf = f_in.read()
        assert not buf
    if not keep:
        os.remove(input_file)
        if not quiet:
            print(f"Removed {input_file}")

def save_jsonl(iterable, output_file, compression=None, processes=64, size_hint=None, overwrite=True, yes=True):
    compression = _determine_compression(output_file, compression)
    f = None
    part = 0
    for item in tqdm(iterable, desc="Writing to JSONL", unit="sample"):
        if f is None:
            part_file = output_file.format(part=part)
            check_arguments(part_file, overwrite, yes)
            f = _open_jsonl(part_file, mode="wb", compression=compression, processes=processes)
        f.write(f"{json.dumps(item)}\n".encode("utf-8"))
        if size_hint is not None and f.tell() >= size_hint:
            f.close()
            part += 1
            f = None
    if f is not None:
        f.close()

def save_mds(it, output_dir, processes=64, compression=None, buf_size=2**24, pigz=False, shard_size=None, size_hint=None, overwrite=True, yes=True):
    if compression == "none" or pigz:
        compression = None
    if compression == "gzip":
        compression = "gz"
    writer = None
    part = 0
    files = []
    for sample in tqdm(it, desc="Writing to MDS", unit="sample"):
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
            for file_name in tqdm(file_names, desc="Compressing with pigz", unit="file"):
                compressed_file_name = file_name + ".gz"
                file_path = os.path.join(output_dir, file_name)
                compressed_file_path = os.path.join(output_dir, compressed_file_name)
                _pigz_compress(file_path, compressed_file_path, processes, buf_size=buf_size, keep=False, quiet=True)
                name2info[file_name]["compression"] = "gz"
                name2info[file_name]["zip_data"] = {
                    "basename": compressed_file_name,
                    "bytes": os.stat(compressed_file_path).st_size,
                    "hashes": {},
                }
            json.dump(index, open(index_path, "wt"))
            print(f"Compressed {output_dir} with pigz")

def save_parquet(it, output_file, compression=None, batch_size=2**16, size_hint=None, overwrite=True, yes=True):
    writer = None
    part = 0
    it = tqdm(it, desc="Writing to Parquet", unit="sample")
    for batch in batch_iterable(it, batch_size):
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

def use_pigz(compression):
    """Determine if pigz should be used based on the compression type."""
    return compression == "pigz" or (compression == "gzip" and _pigz_available())
