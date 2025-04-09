import click

from .jsonl import jsonl

__all__ = ["mds"]

@click.group()
def mds():
    pass

mds.add_command(jsonl)
