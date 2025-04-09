import click

from .mds import mds
from .parquet import parquet

__all__ = ["jsonl"]

@click.group()
def jsonl():
    pass

jsonl.add_command(mds)
jsonl.add_command(parquet)
