import click

from .jsonl import jsonl
from .parquet import parquet

__all__ = ["convert"]

@click.group()
def convert():
    pass

convert.add_command(parquet)
convert.add_command(jsonl)
