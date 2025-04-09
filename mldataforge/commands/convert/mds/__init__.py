import click

from .jsonl import jsonl
from .parquet import parquet

__all__ = ["mds"]

@click.group()
def mds():
    pass

mds.add_command(jsonl)
mds.add_command(parquet)
