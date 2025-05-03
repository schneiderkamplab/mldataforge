import ast
from pathlib import Path
import re
from typing import Callable

from .lazy_dict import LazyDict

__all__ = ['Transformation', 'Transformations', 'flatten_json', 'get_transformations', 'identity', 'unflatten_json']

class Transformation:
    def __init__(self, code: str | Callable):
        self.code = code
        self._init_context()

    def _init_context(self):
        if callable(self.code):
            self.process = self.code
        else:
            global_context = {}
            exec(self.code, global_context)
            if 'process' not in global_context or not callable(global_context['process']):
                raise ValueError("code must define a callable named 'process'")
            self.process = global_context['process']
        self._flushable = hasattr(self.process, 'flushable') and self.process.flushable

    def _normalize_outputs(self, result):
        if result is None:
            return []
        if isinstance(result, (list, tuple, set)):
            return list(result)
        return [result]

    def _flush(self):
        if self._flushable:
            while True:
                flushed = self._normalize_outputs(self.process(None))
                if not flushed:
                    return
                yield from flushed

    def __call__(self, iterable):
        for sample in iterable:
            results = self._normalize_outputs(self.process(sample))
            yield from results
            if not results:
                yield from self._flush()
        if self._flushable:
            yield from self._flush()

class Transformations:
    def __init__(self, codes: list[str | Callable], indices=None):
        self.pipeline = [Transformation(code) for code in codes]
        self.indices = indices

    def __call__(self, dataset):
        result = dataset
        for transform in self.pipeline:
            result = transform(result)
        return result

def flatten_json(obj, parent_key='', sep='$', escape_char='\\'):
    def escape(key):
        return key.replace(escape_char, escape_char * 2)\
                  .replace(sep, escape_char + sep)\
                  .replace('[', escape_char + '[')\
                  .replace(']', escape_char + ']')
    if isinstance(obj, LazyDict):
        obj = obj.materialize()
    items = []
    if isinstance(obj, dict):
        if not obj:
            # explicitly handle empty dict
            items.append((parent_key, {}))
        else:
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{escape(k)}" if parent_key else escape(k)
                items.extend(flatten_json(v, new_key, sep, escape_char).items())
    elif isinstance(obj, list):
        if not obj:
            # explicitly handle empty list
            items.append((parent_key, []))
        else:
            for idx, v in enumerate(obj):
                new_key = f"{parent_key}[{idx}]"
                items.extend(flatten_json(v, new_key, sep, escape_char).items())
    else:
        items.append((parent_key, obj))
    return dict(items)

def get_transformations(trafo: list[str | Callable]):
    codes = []
    if trafo:
        for t in trafo:
            if Path(t).is_file():
                codes.append(Path(t).read_text())
            else:
                try:
                    ast.parse(t)
                    codes.append(t)
                except SyntaxError:
                    raise ValueError(f"Invalid transformation (neither an existing file nor valid Python code): {t}")
    return Transformations(codes=codes)

def identity(obj):
    return obj

def unflatten_json(flat_dict, sep='$', escape_char='\\'):
    def check_flat_json(obj):
        assert isinstance(obj, dict), "Input must be a dictionary"
        for k, v in obj.items():
            assert isinstance(k, str), f"Key {k} is not a string"
            assert isinstance(v, (str, int, float, bool)), f"Value {v} is not a valid JSON type"
    def parse_key(key):
        tokens = re.findall(r'(?:[^.\[\]\\]|\\.)+|\[\d+\]', key)
        parsed = []
        for token in tokens:
            if token.startswith('['):
                parsed.append(int(token[1:-1]))
            else:
                parsed.append(token.replace(escape_char + sep, sep)
                                  .replace(escape_char + '[', '[')
                                  .replace(escape_char + ']', ']')
                                  .replace(escape_char*2, escape_char))
        return parsed
    if isinstance(flat_dict, LazyDict):
        flat_dict = flat_dict.materialize()
    check_flat_json(flat_dict)
    result = {}
    for compound_key, value in flat_dict.items():
        keys = parse_key(compound_key)
        current = result
        for idx, key in enumerate(keys):
            if idx == len(keys) - 1:
                if isinstance(key, int):
                    if not isinstance(current, list):
                        current_parent[last_key] = []
                        current = current_parent[last_key]
                    while len(current) <= key:
                        current.append(None)
                    current[key] = value
                else:
                    current[key] = value
            else:
                next_key = keys[idx + 1]
                if isinstance(key, int):
                    if not isinstance(current, list):
                        current_parent[last_key] = []
                        current = current_parent[last_key]
                    while len(current) <= key:
                        current.append(None)
                    if current[key] is None:
                        current[key] = [] if isinstance(next_key, int) else {}
                    current_parent = current
                    current = current[key]
                else:
                    if key not in current:
                        current[key] = [] if isinstance(next_key, int) else {}
                    current_parent = current
                    current = current[key]
            last_key = key
    return result
