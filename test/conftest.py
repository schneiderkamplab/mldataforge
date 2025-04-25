from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import math
from mldataforge import utils
from pathlib import Path
import pytest
import shutil

def pytest_collection_modifyitems(config, items):
    name_to_node = {}
    dependency_graph = defaultdict(set)
    for item in items:
        name = None
        depends_on = []
        for mark in item.iter_markers(name="dependency"):
            name = mark.kwargs.get("name")
            depends = mark.kwargs.get("depends", [])
            scope = mark.kwargs.get("scope", "module")
            if scope == "session":
                depends_on.extend(depends)
        if name:
            name_to_node[name] = item
        if depends_on:
            for dep in depends_on:
                dependency_graph[item].add(dep)
    for item, deps in list(dependency_graph.items()):
        resolved_deps = set()
        for dep in deps:
            dep_item = name_to_node.get(dep)
            if dep_item:
                resolved_deps.add(dep_item)
        dependency_graph[item] = resolved_deps
    visited = set()
    ordered = []
    def visit(item):
        if item in visited:
            return
        visited.add(item)
        for dep_item in dependency_graph.get(item, []):
            visit(dep_item)
        ordered.append(item)
    for item in items:
        visit(item)
    items[:] = ordered
import pytest
from functools import wraps

def pytest_addoption(parser):
    parser.addoption("--samples", type=int, default=None, help="Number of samples to test on (default: all)")
    parser.addoption("--tmp-path", type=str, default=None, help="Optional path to use instead of pytest's tmp_path")
    parser.addoption("--progress", action="store_true", default=False, help="Enable progress bar for loading datasets")
    parser.addoption("--compressions", type=str, default=None, help="Comma-separated list of compressions to test (default: all)")

@pytest.fixture(autouse=True)
def disable_progress_bar(request):
    utils.CFG["progress"]= request.config.getoption("--progress")
    utils.CFG["echo"] = False

@pytest.fixture(autouse=True)
def compressions(request):
    compressions = request.config.getoption("--compressions")
    if compressions is not None:
        compressions = compressions.split(",")
    return compressions

@pytest.fixture
def tmp_dir(request, tmp_path_factory):
    custom_path = request.config.getoption("--tmp-path")
    sample_size = request.config.getoption("--samples")

    if custom_path:
        tmp_dir = Path(custom_path)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tmp_path_factory.mktemp("test_data")
    if not (
        (tmp_dir / "test.jsonl").exists() and
        (tmp_dir / "test.size").exists() and
        open(tmp_dir / "test.size", "r").read().strip() == str(sample_size)
    ):
        open(str(tmp_dir / "test.size"), "w").write(f"{sample_size}\n")
        hf_hub_download(
            repo_type="dataset",
            repo_id="jlpang888/tulu_300k",
            revision="main",
            filename="data/train-00000-of-00001.parquet",
            local_dir=tmp_dir,
            cache_dir=tmp_dir,
            force_download=False,
        )
        ds = load_dataset("parquet", data_files=[str(tmp_dir / "data" / "train-00000-of-00001.parquet")], split="train")
        if sample_size:
            ds = utils._limit_iterable(ds, sample_size)
        else:
            sample_size = len(ds)
        utils.save_jsonl(
            ds,
            output_file=str(tmp_dir / "test.jsonl"),
            compression=None,
        )
        shutil.rmtree(tmp_dir / "data")
    assert (tmp_dir / "test.jsonl").exists(), f"File {tmp_dir / 'test.jsonl'} does not exist."    
    assert (tmp_dir / "test.size").exists(), f"File {tmp_dir / 'test.size'} does not exist."
    return tmp_dir

@pytest.fixture
def scale_factor(tmp_dir):
    size = (tmp_dir / "test.jsonl").stat().st_size
    factor = int(2 ** math.ceil(math.log2(size / 2**20)))
    return max(factor, 1)
