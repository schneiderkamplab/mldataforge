import click

from .commands import convert

@click.group()
def cli():
    pass

cli.add_command(convert)

if __name__ == "__main__":
    cli()
