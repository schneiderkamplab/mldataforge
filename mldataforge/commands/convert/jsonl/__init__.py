import click

from .mds import mds

__all__ = ["jsonl"]

@click.group()
def jsonl():
    pass

jsonl.add_command(mds)
