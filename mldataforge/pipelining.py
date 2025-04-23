import click
from pathlib import Path
import itertools

from .indexing import IndexedDatasetView, shuffle_permutation
from .utils import ConcatDataset, load_index, load_jsonl_files, load_mds_directories, load_parquet_files, save_jsonl, save_mds, save_parquet

def run_pipeline(cfg, working_dir="."):
    named_iterators = {}
    defaults = cfg.get("defaults", {})
    if working_dir is not None:
        defaults["working_dir"] = working_dir
    for step in cfg["steps"]:
        run_step(defaults, step, named_iterators)

def run_step(defaults, step, named_iterators):
    ds = load_sources(defaults, step["sources"], named_iterators)
    if "shuffle" in step:
        if "index" in step:
            raise click.BadArgumentUsage("Cannot shuffle and index at the same time")
        indices = shuffle_permutation(len(ds), step["shuffle"])
        ds = IndexedDatasetView(ds, indices=indices)
    elif "index" in step:
        indices = load_index(step["index"])
        ds = IndexedDatasetView(ds, indices=indices)
    trafo = step.get("transformations", None)
    save_sink(defaults, step.get("sink", None), ds, trafo, named_iterators)

def load_sources(defaults, sources, named_iterators):
    iterators = []
    for source in sources:
        fmt = None
        if isinstance(source, dict):
            if "fmt" in source:
                fmt = source["fmt"]
                if "path" in source:
                    path = get_path(defaults, source["path"])
                if "name" in source:
                    name = source["name"]
            elif "name" in source:
                fmt = "named"
                name = source["name"]
            elif "path" in source:
                path = get_path(defaults, source["path"])
                if path.is_dir():
                    fmt = "mds"
                elif path.is_file() and path.suffix.lower() == ".parquet":
                    fmt = "parquet"
                elif path.is_file() and ".jsonl" in (suffix.lower for suffix in path.suffixes):
                    fmt = "jsonl"
        else:
            path = get_path(defaults, source)
            if path.is_dir():
                fmt = "mds"
            elif path.is_file() and path.suffix.lower() == ".parquet":
                fmt = "parquet"
            elif path.is_file() and ".jsonl" in (suffix.lower() for suffix in path.suffixes):
                fmt = "jsonl"
            else:
                fmt = "named"
                name = source
            source = {}
        if fmt == "named":
            if name in named_iterators:
                iterators.append(named_iterators[name])
                del named_iterators[name]
            else:
                raise click.BadArgumentUsage(f"Named iterator '{name}' not found")
        elif fmt == "jsonl":
            path = str(path)
            ds = load_jsonl_files([path])
            iterators.append(ds)
        elif fmt == "mds":
            path = str(path)
            ds = load_mds_directories(
                [path],
                split=source.get("split", '.'),
                batch_size=source.get("batch_size", defaults.get("batch_size", 2**16)),
                bulk=source.get("bulk", defaults.get("bulk", True)),
            )
            iterators.append(ds)
        elif fmt == "parquet":
            path = str(path)
            iterators.append(load_parquet_files([path]))
        else:
            raise click.BadArgumentUsage(f"Unknown source format '{fmt}'")
    if all(hasattr(it, "__len__") for it in iterators):
        return ConcatDataset(iterators)
    return itertools.chain(*iterators)
    

def save_sink(defaults, sink, ds, trafo, named_iterators):
    if sink is None:
        for item in ds:
            pass
        return
    if isinstance(sink, dict):
        if "fmt" in sink:
            fmt = sink["fmt"]
            if "path" in sink:
                path = get_path(defaults, sink["path"])
            if "name" in sink:
                name = sink["name"]
        elif "name" in sink:
            fmt = "named"
            name = sink["name"]
        elif "path" in sink:
            path = get_path(defaults, sink["path"])
            if path.suffix.lower() == ".mds":
                fmt = "mds"
            elif path.suffix.lower() == ".parquet":
                fmt = "parquet"
            elif ".jsonl" in (suffix.lower() for suffix in path.suffixes):
                fmt = "jsonl"
    else:
        path = get_path(defaults, sink)
        if path.suffix.lower() == ".mds":
            fmt = "mds"
        elif path.suffix.lower() == ".parquet":
            fmt = "parquet"
        elif ".jsonl" in (suffix.lower() for suffix in path.suffixes):
            fmt = "jsonl"
        else:
            fmt = "named"
            name = sink
        sink = {}
    if fmt == "named":
        named_iterators[name] = ds
    elif fmt == "jsonl":
        path = str(path)
        save_jsonl(
            ds,
            path,
            compression=sink.get("compression", defaults.get("compression", "infer")),
            compression_args=sink.get("compression_args", defaults.get("compression_args", {"processes": 64})),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
            trafo=trafo,
        )
    elif fmt == "mds":
        path = str(path)
        save_mds(
            ds,
            path,
            compression=sink.get("compression", None),
            compression_args=sink.get("compression_args", defaults.get("compression_args", {"processes": 64})),
            buf_size=sink.get("buffer_size", defaults.get("buffer_size", 2**24)),
            pigz=sink.get("pigz", defaults.get("pigz", True)),
            shard_size=sink.get("shard_size", defaults.get("shard_size", None)),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
            trafo=trafo,
        )
    elif fmt == "parquet":
        path = str(path)
        save_parquet(
            ds,
            path,
            compression=sink.get("compression", None),
            compression_args=sink.get("compression_args", {}),
            batch_size=sink.get("batch_size", defaults.get("batch_size", 2**16)),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
            trafo=trafo,
        )
    else:
        raise click.BadArgumentUsage(f"Unknown sink format '{fmt}'")

def get_path(defaults, path):
    return Path(defaults["working_dir"]) / path
