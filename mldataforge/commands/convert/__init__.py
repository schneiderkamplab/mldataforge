import click

from .parquet import parquet

__all__ = ["convert"]

@click.group()
def convert():
    pass

convert.add_command(parquet)
