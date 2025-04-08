import atexit
import bz2
import click
from datasets import concatenate_datasets, load_dataset
import gzip
import lzma
from mltiming import timing
import os
import shutil
import subprocess
import tempfile
from tqdm import tqdm

__all__ = [
    "check_overwrite",
    "create_temp_file",
    "determine_compression",
    "infer_mds_encoding",
    "infer_compression",
    "load_parquet_files",
    "open_jsonl",
    "pigz_available",
    "pigz_compress",
    "use_pigz",
]

class PigzFile(object):
    """A wrapper for pigz to handle gzip compression and decompression."""
    def __init__(self, path, mode="rt", processes=4, encoding=None):
        if mode not in ("rt", "wt", "rb", "wb"):
            raise ValueError("Mode must be one of rt, wt, rb, or wb.")
        self.path = path
        self.mode = mode
        self.processes = processes
        self.encoding = "latin1" if mode[1] == "b" else ("utf-8" if encoding is None else encoding)
        self._process = None
        self._fw = None
        if self.mode[0] == "r":
            args = ["pigz", "-d", "-c", "-p", str(self.processes), self.path]
            self._process = subprocess.Popen(
                args, 
                stdout=subprocess.PIPE,
                encoding=encoding,
            )
        elif self.mode[0] == "w":
            args = ["pigz", "-p", str(self.processes), "-c"]
            self._fw = open(self.path, "w+")
            self._process = subprocess.Popen(
                args, 
                stdout=self._fw,
                stdin=subprocess.PIPE,
                encoding=encoding,
            )
        
    def __iter__(self):
        assert self.mode[0] == "r"
        for line in self._process.stdout:
            yield line
        self._process.wait()
        assert self._process.returncode == 0
        self._process.stdout.close()
        self._process = None
        
    def write(self, line):
        assert self.mode[0] == "w"
        self._process.stdin.write(line if self.mode[1] == "t" else line.encode(self.encoding))
    
    def close(self): 
        if self._process:
            if self.mode[0] == "r":
                self._process.kill()
                self._process.stdout.close()
                self._process = None
            elif self.mode[1] == "w":
                self._process.stdin.close()
                self._process.wait()
                self._process = None
                self._fw.close()
                self._fw = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def check_overwrite(output_path, overwrite, yes):
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
    extension = os.path.splitext(file_path)[1]
    if extension.endswith('.gz'):
        if pigz_available():
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

def load_parquet_files(parquet_files):
    dss = []
    for parquet_file in tqdm(parquet_files, desc="Loading parquet files", unit="file"):
        ds = load_dataset("parquet", data_files=parquet_file, split="train")
        dss.append(ds)
    if len(dss) == 1:
        ds = dss[0]
    else:
        with timing(message=f"Concatenating {len(dss)} datasets"):
            ds = concatenate_datasets(dsets=dss)
    return ds

def open_jsonl(file_path, mode="rt", compression="infer"):
    """Open a JSONL file, handling gzip compression if necessary."""
    compression = determine_compression(file_path, compression)
    if compression in ("gzip", "pigz"):
        return gzip.open(file_path, mode)
    if compression == "bz2":
        return bz2.open(file_path, mode)
    if compression == "xz":
        return lzma.open(file_path, mode)
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
    with open(input_file, "rt", encoding="latin1") as f_in, PigzFile(output_file, "wb", processes=processes) as f_out:
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
