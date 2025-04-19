import re
from typing import Callable

__all__ = ['Trafo', 'flatten_json', 'unflatten_json']

class Trafo:
    """
    Base class for transformations.
    """

    def __init__(self, trafo: Callable | str | None):
        self.trafo = trafo
        if isinstance(trafo, str):
            self.trafo = eval(trafo)

    def __call__(self, obj):
        return self.trafo(obj) if self.trafo else obj

    def __repr__(self):
        return f"{self.__class__.__name__}({self.trafo})"


def flatten_json(obj, parent_key='', sep='.', escape_char='\\'):
    items = []

    def escape(key):
        return key.replace(escape_char, escape_char * 2)\
                  .replace(sep, escape_char + sep)\
                  .replace('[', escape_char + '[')\
                  .replace(']', escape_char + ']')

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


def unflatten_json(flat_dict, sep='.', escape_char='\\'):

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
