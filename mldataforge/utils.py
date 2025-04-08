import atexit
import bz2
import click
import gzip
import lzma
import os
import tempfile

__all__ = [
    "confirm_overwrite",
    "create_temp_file",
    "infer_mds_encoding",
    "infer_compression",
    "open_jsonl",
]

def confirm_overwrite(message):
    print(message)
    response = input("Are you sure you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        raise click.Abort()

def create_temp_file():
    def _cleanup_file(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass
    # Create a named temp file, don't delete right away
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp_name = temp.name
    # Close so others can open it again without conflicts (especially on Windows)
    temp.close()

    # Schedule its deletion at exit
    atexit.register(_cleanup_file, temp_name)

    return temp_name

def determine_compression(file_path, compression="infer"):
    if compression == "infer":
        compression = infer_compression(file_path)
    if compression == "none":
        compression = None
    return compression

def infer_mds_encoding(value):
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

def infer_compression(file_path):
    """Infer the compression type from the file extension."""
    if file_path.lower().endswith('.gz'):
        return 'gzip'
    elif file_path.lower().endswith('.bz2'):
        return 'bz2'
    elif file_path.lower().endswith('.xz'):
        return 'xz'
    elif file_path.lower().endswith('.zip'):
        return 'zip'
    elif file_path.lower().endswith('.zst'):
        return 'zstd'
    else:
        return None

def open_jsonl(file_path, mode="rt", compression="infer"):
    """Open a JSONL file, handling gzip compression if necessary."""
    compression = determine_compression(file_path, compression)
    if compression == "gzip":
        return gzip.open(file_path, mode)
    if compression == "bz2":
        return bz2.open(file_path, mode)
    if compression == "xz":
        return lzma.open(file_path, mode)
    if compression is None:
        return open(file_path, mode)
    raise ValueError(f"Unsupported compression type: {compression}")
