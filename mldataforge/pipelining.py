import click
from pathlib import Path
import itertools

from .indexing import *
from .trafos import *
from .utils import *

def run_pipeline(cfg, working_dir="."):
    named_iterators = {}
    defaults = cfg.get("defaults", {})
    if working_dir is not None:
        defaults["working_dir"] = working_dir
    for step in cfg["steps"]:
        run_step(defaults, step, named_iterators)

def _exclusive_keys(step, keys):
    for key1, key2 in itertools.combinations(keys, 2):
        if step.get(key1, None) is not None and step.get(key2, None) is not None:
            raise click.BadArgumentUsage(f"Cannot use both '{key1}' and '{key2}' at the same time")

def _maybe_reorder(ds, step):
    shuffle = step.get("shuffle", None)
    index = step.get("index", None)
    sort_key = step.get("sort_key", None)
    if shuffle is None and index is None and sort_key is None:
        return ds
    if not isinstance(ds, ConcatDataset):
        raise click.BadArgumentUsage("Reordering can only be applied to a concatenated dataset")
    _exclusive_keys(step, ("shuffle", "index", "sort_key"))
    if shuffle is not None:
        indices = shuffle_permutation(len(ds), seed=shuffle)
        return IndexedDatasetView(ds, indices=indices)
    if index is not None:
        indices = load_index(step["index"])
        return IndexedDatasetView(ds, indices=indices)
    if sort_key is not None:
        indices = sort_permutation(ds, step["sort_key"])
        return IndexedDatasetView(ds, indices=indices)

def run_step(defaults, step, named_iterators):
    ds = load_sources(defaults, step["sources"], named_iterators)
    ds = _maybe_reorder(ds, step)
    trafo = step.get("transformations", None)
    ds = get_transformations(trafo)(ds)
    save_sink(defaults, step.get("sink", None), ds, named_iterators)

def _infer_format(source, item, exists):
    if (exists and item.is_dir()) or item.suffix.lower() == ".mds":
        return "mds"
    if (not exists or item.is_file()) and item.suffix.lower() == ".parquet":
        return "parquet"
    if (not exists or item.is_file()) and item.suffix.lower() == ".msgpack":
        return "msgpack"
    if (not exists or item.is_file()) and item.suffix.lower() == ".jinx":
        return "jinx"
    if (not exists or item.is_file()) and item.suffix.lower() == ".jsonl":
        return "jsonl"
    if (not exists or item.is_file()) and len(item.suffixes) > 1 and item.suffixes[-2].lower() == ".msgpack":
        return "msgpack"
    if (not exists or item.is_file()) and len(item.suffixes) > 1 and item.suffixes[-2].lower() == ".jsonl":
        return "jsonl"
    if exists:
        raise click.BadArgumentUsage(f"Cannot resolve item '{item}' from source '{source}'")
    return "named"

def _resolve_sources(defaults, sources):
    resolved = []
    working_dir = Path(defaults["working_dir"])
    for source in sources:
        if isinstance(source, dict):
            if "name" in source:
                source["fmt"] = "named"
                resolved.append(source)
                continue
            if "path" in source:
                path = source["path"]
                found = list(working_dir.glob(path))
                if not found:
                    raise click.BadArgumentUsage(f"Cannot find any matches for source '{source}'")
                for item in found:
                    res = source.copy()
                    if not "fmt" in source:
                        res["fmt"] = _infer_format(source, item, exists=True)
                    res["path"] = str(item)
                    resolved.append(res)
                continue
            else:
                raise click.BadArgumentUsage(f"Cannot resolve source '{source}'")
        if isinstance(source, str):
            found = list(working_dir.glob(source))
            if not found:
                resolved.append({"fmt": "named", "name": source})
                continue
            for item in found:
                resolved.append({"fmt": _infer_format(source, item, exists=True), "path": str(item)})
    return resolved

def load_sources(defaults, sources, named_iterators):
    iterators = []
    for source in _resolve_sources(defaults, sources):
        fmt = source["fmt"]
        if fmt == "named":
            name = source["name"]
            if name in named_iterators:
                iterators.append(named_iterators[name])
                del named_iterators[name]
                continue
            raise click.BadArgumentUsage(f"Named iterator '{name}' not found")
        path = str(source["path"])
        if fmt == "jinx":
            iterators.append(load_jinx_paths([path]))
        elif fmt == "jsonl":
            iterators.append(load_jsonl_files([path]))
        elif fmt == "mds":
            ds = load_mds_directories(
                [path],
                split=source.get("split", '.'),
                batch_size=source.get("batch_size", defaults.get("batch_size", 2**16)),
                reader=source.get("reader", defaults.get("reader", "ram")),
            )
            iterators.append(ds)
        elif fmt == "msgpack":
            iterators.append(load_msgpack_files([path]))
        elif fmt == "parquet":
            iterators.append(load_parquet_files([path]))
        else:
            raise click.BadArgumentUsage(f"Unknown source format '{fmt}'")
    if all(hasattr(it, "__len__") for it in iterators):
        return ConcatDataset(iterators)
    return itertools.chain(*iterators)

def _resolve_sink(defaults, sink):
    working_dir = Path(defaults["working_dir"])
    if isinstance(sink, dict):
        if "name" in sink:
            sink["fmt"] = "named"
            return sink
        if "path" in sink:
            if not "fmt" in sink:
                sink["fmt"] = _infer_format(sink, working_dir / sink["path"], exists=False)
            sink["path"] = str(working_dir / sink["path"])
            return sink
        raise click.BadArgumentUsage(f"Cannot resolve sink '{sink}'")
    if isinstance(sink, str):
        fmt = _infer_format(sink, working_dir / sink, exists=False)
        if fmt == "named":
            return {"fmt": fmt, "name": sink}
        return {"fmt": fmt, "path": str(working_dir / sink)}
    raise click.BadArgumentUsage(f"Cannot resolve sink '{sink}'")

def save_sink(defaults, sink, ds, named_iterators):
    if sink is None:
        for item in ds:
            pass
        return
    sink = _resolve_sink(defaults, sink)
    fmt = sink["fmt"]
    if fmt == "named":
        name = sink["name"]
        named_iterators[name] = ds
    elif fmt == "jinx":
        path = sink["path"]
        save_jinx(
            ds,
            path,
            compression=sink.get("compression", defaults.get("compression", "snappy")),
            compression_args=sink.get("compression_args", defaults.get("compression_args", {"processes": 64})),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
            compress_threshold=sink.get("compress_threshold", defaults.get("compress_threshold", 2**6)),
            compress_ratio=sink.get("compress_ratio", defaults.get("compress_ratio", 1.0)),
            encoding=sink.get("encoding", defaults.get("encoding", "a85")),
            binary_threshold=sink.get("binary_threshold", defaults.get("binary_threshold", 2**8)),
            ext_sep=sink.get("ext_sep", defaults.get("ext_sep", ".")),
        )
    elif fmt == "jsonl":
        path = sink["path"]
        save_jsonl(
            ds,
            path,
            compression=sink.get("compression", defaults.get("compression", "infer")),
            compression_args=sink.get("compression_args", defaults.get("compression_args", {"processes": 64})),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
        )
    elif fmt == "mds":
        path = sink["path"]
        save_mds(
            ds,
            path,
            compression=sink.get("compression", defaults.get("compression", None)),
            compression_args=sink.get("compression_args", defaults.get("compression_args", {"processes": 64})),
            buf_size=sink.get("buffer_size", defaults.get("buffer_size", 2**24)),
            pigz=sink.get("pigz", defaults.get("pigz", True)),
            shard_size=sink.get("shard_size", defaults.get("shard_size", None)),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
        )
    elif fmt == "msgpack":
        path = sink["path"]
        save_msgpack(
            ds,
            path,
            compression=sink.get("compression", defaults.get("compression", "infer")),
            compression_args=sink.get("compression_args", defaults.get("compression_args", {"processes": 64})),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
        )
    elif fmt == "parquet":
        path = sink["path"]
        save_parquet(
            ds,
            path,
            compression=sink.get("compression", None),
            compression_args=sink.get("compression_args", {}),
            batch_size=sink.get("batch_size", defaults.get("batch_size", 2**16)),
            size_hint=sink.get("size_hint", defaults.get("size_hint", None)),
            overwrite=sink.get("overwrite", defaults.get("overwrite", False)),
            yes=sink.get("yes", defaults.get("yes", False)),
        )
    else:
        raise click.BadArgumentUsage(f"Unknown sink format '{fmt}'")
