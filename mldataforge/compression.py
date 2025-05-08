import bz2
import brotli
import click
import inspect
import io
from isal import igzip as gzip
import lz4
import lzma
import os
import shutil
import snappy
import zstandard

from .brotli import brotli_open
from .pigz import pigz_open
from .snappy import snappy_open

__all__ = [
    "JINX_COMPRESSIONS",
    "JSONL_COMPRESSIONS",
    "MDS_COMPRESSIONS",
    "MSGPACK_COMPRESSIONS",
    "PARQUET_COMPRESSIONS",
    "compress_data",
    "decompress_data",
    "decompress_file",
    "determine_compression",
    "extension_compression",
    "infer_compression",
    "open_compression",
    "pigz_available",
    "pigz_compress",
    "use_pigz",
]

JINX_COMPRESSIONS = dict(
    default=None,
    choices=["none", "brotli", "bz2", "gzip", "lz4", "lzma", "pigz", "snappy", "xz", "zstd"],
)
JSONL_COMPRESSIONS = dict(
    default="infer",
    choices=["infer", "none", "brotli", "bz2", "gzip", "lz4", "lzma", "pigz", "snappy", "xz", "zstd"],
)
MDS_COMPRESSIONS = dict(
    default=None,
    choices=["none", "brotli", "bz2", "gzip", "pigz", "snappy", "zstd", "sample::brotli", "sample::bz2", "sample::gzip", "sample::snappy", "sample::zstd"],
)
MSGPACK_COMPRESSIONS = dict(
    default="infer",
    choices=["infer", "none", "brotli", "bz2", "gzip", "lz4", "lzma", "pigz", "snappy", "xz", "zstd"],
)
PARQUET_COMPRESSIONS = dict(
    default="snappy",
    choices=["snappy", "brotli", "gzip", "lz4", "zstd"],
)

def compress_data(data, ext, chunk_size=65536):
    if ext is None or ext == "none":
        return data

    output = io.BytesIO()

    if ext == "zst":
        cctx = zstandard.ZstdCompressor(level=1)
        with cctx.stream_writer(output, closefd=False) as compressor:
            for i in range(0, len(data), chunk_size):
                compressor.write(data[i:i+chunk_size])

    elif ext == "bz2":
        compressor = bz2.BZ2Compressor()
        for i in range(0, len(data), chunk_size):
            output.write(compressor.compress(data[i:i+chunk_size]))
        output.write(compressor.flush())

    elif ext in ("lzma", "xz"):
        compressor = lzma.LZMACompressor()
        for i in range(0, len(data), chunk_size):
            output.write(compressor.compress(data[i:i+chunk_size]))
        output.write(compressor.flush())

    elif ext == "lz4":
        compressed = lz4.frame.compress(data)
        output.write(compressed)

    elif ext == "snappy":
        compressor = snappy.StreamCompressor()
        for i in range(0, len(data), chunk_size):
            output.write(compressor.compress(data[i:i+chunk_size]))
        output.write(compressor.flush())

    elif ext == "gz":
        with gzip.GzipFile(fileobj=output, mode="wb") as compressor:
            for i in range(0, len(data), chunk_size):
                compressor.write(data[i:i+chunk_size])

    elif ext == "br":
        compressed = brotli.compress(data)
        output.write(compressed)

    else:
        raise ValueError(f"Unsupported compression extension: {ext}")

    return output.getvalue()

def decompress_data(data, ext, chunk_size=65536):
    input_io = io.BytesIO(data)
    output = io.BytesIO()
    if ext == "zst":
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(input_io) as reader:
            while chunk := reader.read(chunk_size):
                output.write(chunk)
    elif ext == "bz2":
        decompressor = bz2.BZ2Decompressor()
        while chunk := input_io.read(chunk_size):
            output.write(decompressor.decompress(chunk))
    elif ext in ("lzma", "xz"):
        decompressor = lzma.LZMADecompressor()
        while chunk := input_io.read(chunk_size):
            output.write(decompressor.decompress(chunk))
    elif ext == "lz4":
        with lz4.frame.open(input_io, "rb") as reader:
            while chunk := reader.read(chunk_size):
                output.write(chunk)
    elif ext == "snappy":
        decompressor = snappy.StreamDecompressor()
        while chunk := input_io.read(chunk_size):
            output.write(decompressor.decompress(chunk))
    elif ext == "gz":
        with gzip.GzipFile(fileobj=input_io, mode="rb") as reader:
            while chunk := reader.read(chunk_size):
                output.write(chunk)
    elif ext == "br":
        # Brotli python API does not have streaming decompression, fallback to full decompress
        decompressed = brotli.decompress(input_io.read())
        output.write(decompressed)
    else:
        raise ValueError(f"Unsupported compression extension: {ext}")
    return output.getvalue()

def decompress_file(input_path, output_path, ext, chunk_size=65536):
    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        if ext == "zst":
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(f_in) as reader:
                while chunk := reader.read(chunk_size):
                    f_out.write(chunk)
        elif ext == "bz2":
            d = bz2.BZ2Decompressor()
            while chunk := f_in.read(chunk_size):
                f_out.write(d.decompress(chunk))
        elif ext in ("lzma", "xz"):
            d = lzma.LZMADecompressor()
            while chunk := f_in.read(chunk_size):
                f_out.write(d.decompress(chunk))
        elif ext == "lz4":
            with lz4.frame.open(f_in, "rb") as reader:
                while chunk := reader.read(chunk_size):
                    f_out.write(chunk)
        elif ext == "snappy":
            # Read full file, decompress at once
            compressed = f_in.read()
            decompressed = snappy.StreamDecompressor().decompress(compressed)
            f_out.write(decompressed)
        elif ext == "gz":
            with gzip.open(f_in, "rb") as reader:
                while chunk := reader.read(chunk_size):
                    f_out.write(chunk)
        elif ext == "br":
            # Read full file, decompress at once
            compressed = f_in.read()
            decompressed = brotli.decompress(compressed)
            f_out.write(decompressed)
        else:
            raise ValueError(f"Unsupported compression extension: {ext}")

def determine_compression(fmt, file_path, compression="infer", no_pigz=False):
    if compression == "none":
        return None
    if fmt == "jinx":
        if compression == "brotli":
            return "br"
        if compression == "gzip":
            return "gz"
        if compression == "zstd":
            return "zst"
        return compression
    if fmt in ("jsonl", "msgpack"):
        if compression == "infer":
            compression = infer_compression(file_path)
        if compression == "brotli":
            return "br"
        return compression
    if fmt == "mds":
        if compression == "infer":
            raise ValueError()
        if compression == "pigz" or (not no_pigz and compression == "gzip" and pigz_available()):
            return None
        if compression == "gzip":
            return "gz"
        if compression == "brotli":
            return "br"
        if compression == "sample::gzip":
            return "sample::gz"
        if compression == "sample::brotli":
            return "sample::br"
        return compression
    if fmt == "parquet":
        return compression
    raise ValueError(f"Unsupported format: {format}")

def extension_compression(compression, file_path):
    """Get the file extension for the given compression type."""
    if compression == "infer":
        compression = infer_compression(file_path)
    if compression == "brotli":
        return ".br"
    if compression == "bz2":
        return ".bz2"
    if compression in ("gzip", "pigz"):
        return ".gz"
    if compression == "lz4":
        return ".lz4"
    if compression == "lzma":
        return ".lzma"
    if compression == "snappy":
        return ".snappy"
    if compression == "xz":
        return ".xz"
    if compression == "zstd":
        return ".zst"
    if compression is None or compression == "none":
        return ""
    raise ValueError(f"Unsupported compression type: {compression}")

def infer_compression(file_path, pigz=True):
    """Infer the compression type from the file extension."""
    extension = os.path.splitext(file_path)[1]
    if extension.endswith('.br'):
        return 'brotli'
    if extension.endswith('.bz2'):
        return 'bz2'
    if extension.endswith('.gz'):
        if pigz and pigz_available():
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

def with_kwargs(func, candidates, *args, **kwargs):
    params = inspect.signature(func).parameters
    filtered = (candidates.copy() if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()) else {k: v for k, v in candidates.items() if k in params and params[k].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)})
    filtered.update(kwargs)
    return func(*args, **filtered)

def open_compression(file_path, mode="rt", compression="infer", compression_args={"processes": 64}):
    """Open a file, handling compression if necessary."""
    if compression == "infer":
        compression = infer_compression(file_path)
    if compression in ("brotli", "br"):
        return with_kwargs(brotli_open, compression_args, file_path, mode)
    if compression in ("gzip", "gz"):
        return with_kwargs(gzip.open, compression_args, file_path, mode)
    if compression == "pigz":
        return with_kwargs(pigz_open if mode[0] == "w" else gzip.open, compression_args, file_path, mode)
    if compression == "bz2":
        return with_kwargs(bz2.open, compression_args, file_path, mode)
    if compression == "lz4":
        return with_kwargs(lz4.frame.open, compression_args, file_path, mode)
    if compression in ("lzma", "xz"):
        return with_kwargs(lzma.open, compression_args, file_path, mode)
    if compression == "snappy":
        return with_kwargs(snappy_open, compression_args, file_path, mode)
    if compression == "zstd":
        return with_kwargs(zstandard.open, compression_args, file_path, mode)
    if compression is None or compression == "none":
        return with_kwargs(open, compression_args, file_path, mode)
    raise ValueError(f"Unsupported compression type: {compression}")

def pigz_available():
    """Check if pigz is available on the system."""
    return shutil.which("pigz") is not None

def pigz_compress(input_file, output_file, processes=64, buf_size=2**24, keep=False, quiet=False):
    """Compress a file using pigz."""
    with open(input_file, "rb") as f_in, pigz_open(output_file, "wb", processes=processes) as f_out:
        while True:
            buf = f_in.read(buf_size)
            if not buf:
                break
            f_out.write(buf)
    if not keep:
        os.remove(input_file)
        if not quiet:
            click.echo(f"Removed {input_file}")

def use_pigz(compression, no_pigz=False):
    """Determine if pigz should be used based on the compression type."""
    return compression == "pigz" or (not no_pigz and compression == "gzip" and pigz_available())
