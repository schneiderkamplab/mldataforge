import bz2
from isal import igzip as gzip
import lz4
import lzma
import os
import shutil
import snappy
from tqdm import tqdm
import zipfile
import zstandard

from .pigz import pigz_open

__all__ = [
    "JSONL_COMPRESSIONS",
    "MDS_COMPRESSIONS",
    "PARQUET_COMPRESSIONS",
    "determine_compression",
    "extension_compression",
    "infer_compression",
    "open_compression",
    "pigz_available",
    "pigz_compress",
    "use_pigz",
]

JSONL_COMPRESSIONS = ["infer", "none", "bz2", "gzip", "lz4", "lzma", "pigz", "snappy", "xz", "zip", "zstd"]
MDS_COMPRESSIONS = [None, "none", "brotli", "bz2", "gzip", "pigz", "snappy", "zstd"]
PARQUET_COMPRESSIONS = ["snappy", "brotli", "gzip", "lz4", "zstd"]

def determine_compression(file_path, compression="infer"):
    if compression == "infer":
        compression = infer_compression(file_path)
    if compression == "none":
        compression = None
    return compression

def extension_compression(compression, file_path):
    """Get the file extension for the given compression type."""
    if compression == "infer":
        compression = infer_compression(file_path)
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

def infer_compression(file_path):
    """Infer the compression type from the file extension."""
    extension = os.path.splitext(file_path)[1]
    if extension.endswith('.bz2'):
        return 'bz2'
    if extension.endswith('.gz'):
        if pigz_available():
            return 'pigz'
        return 'gzip'
    if extension.endswith('.lz4'):
        return 'lz4'
    if extension.endswith('.lzma'):
        return 'lzma'
    if extension.endswith('.snappy'):
        return 'snappy'
    if extension.endswith('.xz'):
        return 'xz'
    if extension.endswith('.zip'):
        return 'zip'
    if extension.endswith('.zst'):
        return 'zstd'
    return None

def open_compression(file_path, mode="rt", compression="infer", processes=64):
    """Open a file, handling gzip compression if necessary."""
    compression = determine_compression(file_path, compression)
    if compression == "gzip":
        return gzip.open(file_path, mode)
    if compression == "pigz":
        return pigz_open(file_path, mode, processes=processes) if mode[0] == "w" else gzip.open(file_path, mode)
    if compression == "bz2":
        return bz2.open(file_path, mode)
    if compression == "lz4":
        return lz4.frame.open(file_path, mode)
    if compression in ("lzma", "xz"):
        return lzma.open(file_path, mode)
    if compression == "snappy":
        return snappy.open(file_path, mode)
    if compression == "zip":
        return zipfile.ZipFile(file_path, mode)
    if compression == "zstd":
        return zstandard.open(file_path, mode)
    if compression is None:
        return open(file_path, mode)
    raise ValueError(f"Unsupported compression type: {compression}")

def pigz_available():
    """Check if pigz is available on the system."""
    return shutil.which("pigz") is not None

def pigz_compress(input_file, output_file, processes=64, buf_size=2**24, keep=False, quiet=False):
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

def use_pigz(compression):
    """Determine if pigz should be used based on the compression type."""
    return compression == "pigz" or (compression == "gzip" and pigz_available())
