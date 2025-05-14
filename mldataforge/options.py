import click

from .compression import JSONL_COMPRESSIONS, MDS_COMPRESSIONS, MSGPACK_COMPRESSIONS, PARQUET_COMPRESSIONS
from .mds import MDS_READERS

__all__ = [
    "batch_size_option",
    "binary_threshold_option",
    "buf_size_option",
    "compress_threshold_option",
    "compress_ratio_option",
    "compression_args_option",
    "compression_option",
    "encoding_option",
    "every_option",
    "ext_sep_option",
    "index_option",
    "lazy_option",
    "mmap_option",
    "no_pigz_option",
    "number_option",
    "offset_option",
    "output_dir_option",
    "override_encoding_option",
    "overwrite_option",
    "prefix_option",
    "reader_option",
    "shard_size_option",
    "shuffle_option",
    "size_hint_option",
    "sort_key_option",
    "split_option",
    "trafo_option",
    "yes_option",
]

def batch_size_option(default=2**16):
    """
    Option for specifying the batch size.
    """
    return click.option(
        "--batch-size",
        default=default,
        help=f"Batch size for loading data and writing files (default: {default}).",
    )

def binary_threshold_option(default=None):
    """
    Option for specifying the binary threshold.
    """
    return click.option(
        "--binary-threshold",
        default=default,
        type=int,
        help=f"Binary threshold for compression (default: {default}).",
    )

def buf_size_option(default=2**24):
    """
    Option for specifying the buffer size.
    """
    return click.option(
        "--buf-size",
        default=default,
        help=f"Buffer size for pigz compression (default: {default}).",
    )

def compress_threshold_option(default=2**6):
    """
    Option for specifying the compression threshold under which to not compress.
    """
    return click.option(
        "--compress-threshold",
        default=default,
        type=int,
        help=f"Compression threshold (default: {default}).",
    )

def compress_ratio_option(default=1.0):
    """
    Option for specifying the worst compression ratio allowed.
    """
    return click.option(
        "--compress-ratio",
        default=default,
        type=float,
        help=f"Compression ratio (default: {default}).",
    )

def compression_args_option(default={"processes": 64}):
    """
    Option for specifying compression arguments.
    """
    return click.option(
        "--compression-args",
        default=default,
        type=dict,
        help=f"Compression arguments (default: {default}).",
    )

def compression_option(args):
    """
    Option for specifying the compression type.
    """
    return click.option(
        "--compression",
        default=args["default"],
        type=click.Choice(args["choices"], case_sensitive=False),
        help=f'Compress the output file (default: {args["default"]}).',
    )

def encoding_option(default="a85"):
    """
    Option for specifying the encoding type.
    """
    return click.option(
        "--encoding",
        default=default,
        type=click.Choice(["a85", "b64", "hex"], case_sensitive=False),
        help=f"Encoding type (default: {default}).",
    )

def every_option(default=None):
    """
    Option for specifying the frequency of processing items.
    """
    return click.option(
        "--every",
        default=default,
        help="Process every N-th item (default: {default}).",
    )

def ext_sep_option(default="."):
    """
    Option for specifying the extension separator.
    """
    return click.option(
        "--ext-sep",
        default=default,
        help=f"Extension separator (default: {default}).",
    )

def index_option():
    """
    Option for specifying an index file.
    """
    return click.option(
        "--index",
        default=None,
        type=click.Path(exists=True),
        help="Index file for loading the dataset.",
    )

def lazy_option():
    """
    Option for specifying whether to load the dataset lazily.
    """
    return click.option(
        "--lazy",
        is_flag=True,
        help="Load the dataset lazily.",
    )

def mmap_option():
    """
    Option for specifying whether to use memory mapping.
    """
    return click.option(
        "--mmap",
        is_flag=True,
        help="Use memory mapping for loading the dataset.",
    )

def no_pigz_option():
    """
    Option for specifying whether to use pigz compression.
    """
    return click.option(
        "--no-pigz",
        is_flag=True,
        help="Do not use pigz compression.",
    )

def number_option(default=None):
    """
    Option for specifying the number of items to process.
    """
    return click.option(
        "--number",
        default=default,
        type=int,
        help=f"Number of items to process (default: all).",
    )

def offset_option(default=None):
    """
    Option for specifying the offset for processing items.
    """
    return click.option(
        "--offset",
        default=default,
        help=f"Offset for processing items (default: {default}).",
    )

def output_dir_option(default="."):
    """
    Option for specifying the output directory.
    """
    return click.option(
        "--output-dir",
        default=default,
        type=click.Path(exists=False),
        help="Output directory.",
    )

def override_encoding_option(default=None):
    """
    Option for specifying whether to override the encoding.
    """
    return click.option(
        "--override-encoding",
        default=default,
        help="Override the encoding when loading.",
    )

def overwrite_option():
    """
    Option for specifying whether to overwrite existing files.
    """
    return click.option(
        "--overwrite",
        is_flag=True,
        help="Overwrite existing path.",
    )

def prefix_option(default="part-"):
    """
    Option for specifying the prefix for output files.
    """
    return click.option(
        "--prefix",
        default=default,
        help=f"Prefix for output files (default: {default}).",
    )

def reader_option(default=MDS_READERS["default"]):
    """
    Option for specifying the MDS reader type.
    """
    return click.option(
        "--reader",
        default=default,
        type=click.Choice(MDS_READERS["choices"], case_sensitive=False),
        help=f"Reader type (default: {default}).",
    )

def shard_size_option(default=2**26):
    """
    Option for specifying the shard size.
    """
    return click.option(
        "--shard-size",
        default=default,
        type=int,
        help=f"Shard size for the dataset (default: {default}).",
    )

def shuffle_option():
    """
    Option for specifying whether to shuffle the dataset by providing a random seed.
    """
    return click.option(
        "--shuffle",
        default=None,
        type=int,
        help="Shuffle the dataset by providing a random seed.",
    )

def size_hint_option(default=2**26):
    """
    Option for specifying the size hint.
    """
    return click.option(
        "--size-hint",
        default=default,
        help=f"Size hint for the dataset (default: {default}).",
    )

def sort_key_option():
    """
    Option for specifying the sort key.
    """
    return click.option(
        "--sort-key",
        default=None,
        type=str,
        help="Sort key function for the dataset.",
    )

def split_option(default=None):
    """
    Option for specifying the split to use.
    """
    return click.option(
        "--split",
        default=default,
        type=str,
        help="Split to use for the dataset (default: {default}).",
    )

def trafo_option():
    """
    Option for specifying the transformation function.
    """
    return click.option(
        "--trafo",
        default=None,
        type=str,
        help="Transformation function to apply to the dataset.",
        multiple=True,
    )

def yes_option():
    """
    Option for specifying whether to assume yes to all prompts.
    """
    return click.option(
        "--yes",
        is_flag=True,
        help="Assume yes to all prompts. Use with caution as it will remove files or even entire directories without confirmation.",
    )
