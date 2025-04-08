import click

from .jsonl import jsonl
from .mds import mds

__all__ = ["parquet"]

@click.group()
def parquet():
    pass

parquet.add_command(jsonl)
parquet.add_command(mds)
