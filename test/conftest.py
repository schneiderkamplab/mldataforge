from huggingface_hub import hf_hub_download
from pathlib import Path
import pytest

@pytest.fixture
def tmp_dir():
    tmp_dir = Path(__file__).resolve().parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    if not (tmp_dir / "test.parquet").exists():
        hf_hub_download(
            repo_type="dataset",
            repo_id="jlpang888/tulu_300k",
            revision="main",
            filename="data/train-00000-of-00001.parquet",
            local_dir=tmp_dir,
        )
        (tmp_dir / "data/train-00000-of-00001.parquet").rename(
            tmp_dir / "test.parquet"
        )
    return tmp_dir