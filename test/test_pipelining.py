from mldataforge.pipelining import run_pipeline
from mldataforge.utils import load_pipeline_config
from pathlib import Path
import pytest

@pytest.mark.dependency(depends=["conversion"])
@pytest.mark.dependency(name="pipelining")
@pytest.mark.parametrize("config_file", [
    str(Path(__file__).parent / "test.yaml"),
])
def test_pipelining(config_file, tmp_dir):
    cfg = load_pipeline_config(config_file)
    run_pipeline(cfg, working_dir=str(tmp_dir))
    assert (tmp_dir / "test.pipelined.mds").exists()
    assert (tmp_dir / "test.concatenated.mds").exists()
    assert (tmp_dir / "test.sorted.jsonl.gz").exists()
