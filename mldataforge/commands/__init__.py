import click

from .convert import convert
from .index import index
from .join import join
from .pipeline import pipeline
from .split import split

__all__ = ["cli"]

@click.group()
def cli():
    pass

cli.add_command(convert)
cli.add_command(index)
cli.add_command(join)
cli.add_command(pipeline)
cli.add_command(split)
