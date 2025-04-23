import click

from ..pipelining import run_pipeline
from ..utils import load_pipeline_config

@click.command()
@click.argument("pipeline_config", type=click.Path(exists=True))
@click.option("--working-dir", type=click.Path(exists=True), default=".", help="Working directory for the pipeline")
def pipeline(pipeline_config, working_dir):
    cfg = load_pipeline_config(pipeline_config)
    run_pipeline(cfg, working_dir=working_dir)