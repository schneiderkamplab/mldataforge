import click

from .jinx import jinx

__all__ = ["export"]

@click.group()
def export():
    pass

export.add_command(jinx)
