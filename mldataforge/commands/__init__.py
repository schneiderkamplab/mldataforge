import click

from .convert import convert
from .join import join
from .split import split

__all__ = ["cli"]

@click.group()
def cli():
    pass

cli.add_command(convert)
cli.add_command(join)
cli.add_command(split)
