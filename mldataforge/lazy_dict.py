from collections.abc import MutableMapping

__all__ = ["LazyDict"]

class LazyDict(MutableMapping):

    def __init__(self, data, transform_fn, key_fn=lambda x: x, context=None):
        self._transform_fn = transform_fn
        self._key_fn = key_fn
        self.context = context
        self._store = {k: self._wrap_dicts(v) for k, v in data.items()}

    def materialize(self):
        return self._materialize(self)

    def _materialize(self, obj):
        if isinstance(obj, LazyDict):
            return {k: self._materialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._materialize(item) for item in obj]
        elif isinstance(obj, set):
            return {self._materialize(item) for item in obj}
        elif isinstance(obj, tuple):
            return tuple(self._materialize(item) for item in obj)
        else:
            return obj

    def eagerize(self, obj):
        if isinstance(obj, LazyDict):
            return {k: self.eagerize(v) for k, v in obj.raw_items()}
        elif isinstance(obj, dict):
            return {k: self.eagerize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.eagerize(item) for item in obj]
        elif isinstance(obj, set):
            return {self.eagerize(item) for item in obj}
        elif isinstance(obj, tuple):
            return tuple(self.eagerize(item) for item in obj)
        else:
            return obj

    def __getitem__(self, key):
        if key in self._store:
            return self._store[key]

        full_key = self._resolve_full_key(key)
        if full_key is None:
            raise KeyError(f"No key resolved for logical key '{key}'")

        raw_value = self._store[full_key]
        wrapped_value = self._wrap_dicts(raw_value)

        new_value = self._transform_fn(full_key, wrapped_value)

        self._store[key] = new_value
        to_delete = [full_key for full_key in self._store if full_key != key and self._key_fn(full_key) == key]
        for full_key in to_delete:
            del self._store[full_key]
        return new_value

    def _resolve_full_key(self, logical_key):
        for full_key in self._store:
            if self._key_fn(full_key) == logical_key:
                return full_key
        return None

    def _wrap_dicts(self, obj):
        if isinstance(obj, dict):
            return LazyDict(obj, self._transform_fn, self._key_fn, self.context)
        elif isinstance(obj, list):
            return [self._wrap_dicts(item) for item in obj]
        elif isinstance(obj, set):
            return {self._wrap_dicts(item) for item in obj}
        elif isinstance(obj, tuple):
            return tuple(self._wrap_dicts(item) for item in obj)
        else:
            return obj

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        full_key = self._resolve_full_key(key)
        if full_key is None:
            raise KeyError(key)
        del self._store[full_key]

    def __iter__(self):
        logical_keys = {self._key_fn(k) for k in self._store}
        return iter(logical_keys)

    def __len__(self):
        return len(set(self._key_fn(k) for k in self._store))

    def items(self):
        for key in self:
            yield key, self[key]

    def raw_items(self):
        for key in self._store:
            yield key, self._store[key]

    def values(self):
        for key in self:
            yield self[key]

    def __repr__(self):
        return f"LazyDict({self._store})"
