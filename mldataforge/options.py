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

def overwrite_option():
    """
    Option for specifying whether to overwrite existing files.
    """
    return click.option(
        "--overwrite",
        is_flag=True,
        help="Overwrite existing path.",
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

def yes_option():
    """
    Option for specifying whether to assume yes to all prompts.
    """
    return click.option(
        "--yes",
        is_flag=True,
        help="Assume yes to all prompts. Use with caution as it will remove files or even entire directories without confirmation.",
    )
