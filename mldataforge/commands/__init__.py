import click

from .convert import convert
from .join import join

__all__ = ["cli"]

@click.group()
def cli():
    pass

cli.add_command(convert)
cli.add_command(join)
