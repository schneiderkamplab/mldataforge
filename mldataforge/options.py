import click

__alll__ = [
    "batch_size_option",
    "buf_size_option",
    "compression_option",
    "overwrite_option",
    "processes_option",
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

def buf_size_option(default=2**24):
    """
    Option for specifying the buffer size.
    """
    return click.option(
        "--buf-size",
        default=default,
        help=f"Buffer size for pigz compression (default: {default}).",
    )

def no_bulk_option():
    """
    Option for specifying whether to use a custom space and time-efficient bulk reader (only gzip and no compression).
    """
    return click.option(
        "--no-bulk",
        is_flag=True,
        help="Use a custom space and time-efficient bulk reader (only gzip and no compression).",
    )

def compression_option(default, choices):
    """
    Option for specifying the compression type.
    """
    return click.option(
        "--compression",
        default=default,
        type=click.Choice(choices, case_sensitive=False),
        help=f"Compress the output file (default: {default}).",
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

def processes_option(default=64):
    """
    Option for specifying the number of processes to use.
    """
    return click.option(
        "--processes",
        default=default,
        help=f"Number of processes to use (default: {default}).",
    )

def shard_size_option(default=2**26):
    """
    Option for specifying the shard size.
    """
    return click.option(
        "--shard-size",
        default=default,
        help=f"Shard size for the dataset (default: {default}).",
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

def yes_option():
    """
    Option for specifying whether to assume yes to all prompts.
    """
    return click.option(
        "--yes",
        is_flag=True,
        help="Assume yes to all prompts. Use with caution as it will remove files or even entire directories without confirmation.",
    )
