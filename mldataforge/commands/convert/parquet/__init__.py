import click

from .jsonl import jsonl

__all__ = ["parquet"]

@click.group()
def parquet():
    pass

parquet.add_command(jsonl)
