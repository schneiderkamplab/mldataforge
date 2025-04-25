import click

from .jsonl import jsonl
from .mds import mds
from .msgpack import msgpack
from .parquet import parquet

__all__ = ["convert"]

@click.group()
def convert():
    pass

convert.add_command(jsonl)
convert.add_command(mds)
convert.add_command(msgpack)
convert.add_command(parquet)
