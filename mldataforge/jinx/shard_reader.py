import base64
import io
import orjson
import mmap as _mmap
import numpy as np
import os
import tempfile
from pathlib import Path

from ..lazy_dict import LazyDict
from ..compression import decompress_data, decompress_file
from ..encoding import decode_a85_stream_to_file, decode_b64_stream_to_file

__all__ = ["JinxShardReader"]

class JinxShardReader:
    def __init__(self, path: str, split=None, lazy=True, mmap=False, encoding=None):
        self.path = Path(path)
        self.lazy = lazy
        self.mmap = mmap
        self.encoding = encoding
        self.file = self.path.open("rb")
        if self.mmap:
            self.mmap = _mmap.mmap(self.file.fileno(), length=0, access=_mmap.ACCESS_READ)
        self.bin_path = self.path.with_suffix(".binx")
        self.bin = None
        self._load_footer(split=split)

    def _load_footer(self, split=None):
        if self.mmap:
            offset = self.mmap.size()-2
            while self.mmap[offset] != ord("\n"):
                offset -= 1
            footer_offset = int(self.mmap[offset:].decode("utf-8"))
            header_line = self.mmap[footer_offset:offset]
        else:
            self.file.seek(-64, os.SEEK_END)
            last_part = self.file.read()
            lines = last_part.strip().split(b"\n")
            footer_offset = int(lines[-1].decode("utf-8"))
            self.file.seek(footer_offset)
            header_line = self.file.readline().decode("utf-8")
        self.header = orjson.loads(header_line)
        self.num_samples = self.header["num_samples"]
        self.ext_sep = self.header.get("ext_sep", ".")
        self.encoding = self.header.get("encoding", "a85" if self.encoding is None else self.encoding)
        if split is not None and self.header.get("split") != split:
            self.num_samples = 0
        index_key = next((k for k in self.header if k.startswith("index.")), None)
        if index_key is None:
            raise ValueError("Missing index in JINX header.")
        index_data = self.header[index_key]
        extensions = index_key.split(self.ext_sep)[1:]
        self.offsets = self._mmap_index(index_data, extensions)

    def _mmap_index(self, data, extensions):
        if extensions[-1] == "bin":
            location = data
            offset, length = location["offset"], location["length"]
            return np.memmap(self.bin_path, dtype=np.uint64, mode="r", offset=offset, shape=(length // 8))
        tmp_path = tempfile.NamedTemporaryFile(delete=False).name
        if self.encoding == "a85":
            decode_a85_stream_to_file(data, tmp_path)
        elif self.encoding == "b64":
            decode_b64_stream_to_file(data, tmp_path)
        else:
            raise ValueError(f"Unsupported encoding '{self.encoding}' for index data")
        for i, ext in reversed(list(enumerate(extensions))):
            if ext == "npy":
                try:
                    return np.memmap(tmp_path, dtype=np.uint64, mode="r")
                except Exception as e:
                    raise ValueError(f"Failed to load .npy array for key 'index': {e}")
            elif ext in {"zst", "bz2", "lz4", "lzma", "snappy", "xz", "gz", "br"}:
                new_tmp = tempfile.NamedTemporaryFile(delete=False).name
                decompress_file(tmp_path, new_tmp, ext)
                os.remove(tmp_path)
                tmp_path = new_tmp
            else:
                raise ValueError(f"Unsupported compression type in index: {ext}")
        raise ValueError("No valid index extension (like .npy) found")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mmap:
            begin, end = self.offsets[idx:idx+2]
            line = self.mmap[begin:end]
        else:
            if not (0 <= idx < self.num_samples):
                raise IndexError(f"Sample index out of range: {idx}")
            offset = self.offsets[idx]
            self.file.seek(offset)
            line = self.file.readline().decode("utf-8")
        sample = orjson.loads(line)
        return self._load_sample(sample)

    def _lazy_load_value(self, key, value):
        if self.ext_sep not in key:
            return key, value
        parts = key.split(self.ext_sep)
        _, extensions = parts[0], parts[1:]
        decoded, extensions = self._load_bytes(key, value, extensions)
        decoded_value = self._unserialize(key, decoded, extensions, original_value=value)
        return decoded_value

    def _load_value(self, key, value):
        if self.ext_sep not in key:
            return key, value
        parts = key.split(self.ext_sep)
        base_key, extensions = parts[0], parts[1:]
        decoded, extensions = self._load_bytes(key, value, extensions)
        decoded_value = self._unserialize(key, decoded, extensions, original_value=value)
        return base_key, decoded_value

    def _unserialize(self, key, decoded, extensions, original_value=None):
        for ext in reversed(extensions):
            if ext in {"zst", "bz2", "lz4", "lzma", "snappy", "xz", "gz", "br"}:
                try:
                    decoded = decompress_data(decoded, ext)
                except Exception as e:
                    raise ValueError(f"Failed to decompress '{ext}' for key '{key}': {e}")
            elif ext == "npy":
                try:
                    return np.load(io.BytesIO(decoded), allow_pickle=False)
                except Exception as e:
                    raise ValueError(f"Failed to load .npy array for key '{key}': {e}")
            elif ext == "raw":
                return decoded
            elif ext == "np":
                try:
                    dtype_str, buf = decoded.split(b"\x00", 1)
                    return np.frombuffer(buf, dtype=dtype_str.decode("ascii"))[0]
                except Exception as e:
                    raise ValueError(f"Failed to decode np scalar for key '{key}': {e}")
            elif ext == "str":
                try:
                    return decoded.decode("utf-8")
                except Exception as e:
                    raise ValueError(f"Failed to decode UTF-8 string for key '{key}': {e}")
            elif ext in {
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64",
                "float32", "float64"
            }:
                try:
                    return np.dtype(ext).type(original_value)
                except Exception as e:
                    raise ValueError(f"Failed to cast to {ext} for key '{key}': {e}")

            else:
                raise ValueError(f"Unsupported extension '{ext}' for key '{key}'")
        try:
            return orjson.loads(decoded)
        except Exception as e:
            raise ValueError(f"Failed to decode final JSON for key '{key}': {e}")

    def _load_bytes(self, key, value, extensions):
        if extensions[-1] == "bin":
            if not isinstance(value, (dict, LazyDict)) or "offset" not in value:
                raise ValueError(f"Expected offset dict for '.bin' extension in key '{key}'")
            offset, length = value["offset"], value["length"]
            if not self.bin:
                self.bin = open(self.bin_path, "rb")
            self.bin.seek(offset)
            value = self.bin.read(length)
            extensions.pop()
        elif isinstance(value, str):
            if self.encoding == "a85":
                try:
                    value = base64.a85decode(value.encode("ascii"), foldspaces=True)
                except Exception as e:
                    raise ValueError(f"Failed to decode base85 for key '{key}': {e}")
            elif self.encoding == "b64":
                try:
                    value = base64.b64decode(value.encode("ascii"))
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 for key '{key}': {e}")
            elif self.encoding == "hex":
                try:
                    value = bytes.fromhex(value)
                except Exception as e:
                    raise ValueError(f"Failed to decode hex for key '{key}': {e}")
            else:
                raise ValueError(f"Unsupported encoding '{self.encoding}' for key '{key}'")
        return value, extensions

    def _load_sample(self, value):
        if isinstance(value, dict):
            if self.lazy:
                return LazyDict(value, self._lazy_load_value, lambda k: k.split(self.ext_sep, 1)[0], self)
            result = {}
            for k, v in value.items():
                new_key, new_value = self._load_value(k, self._load_sample(v))
                result[new_key] = new_value
            return result
        if isinstance(value, list):
            return [self._load_sample(v) for v in value]
        return value

    def __iter__(self):
        if self.mmap:
            for i in range(self.num_samples):
                begin, end = self.offsets[i:i+2]
                line = self.mmap[begin:end]
                sample = orjson.loads(line)
                yield self._load_sample(sample)
        else:
            original_pos = self.file.tell()
            try:
                if self.offsets[0] > os.path.getsize(self.path):
                    raise ValueError(f"Offset {self.offsets[0]} is larger than file size {os.path.getsize(self.path)}")
                self.file.seek(self.offsets[0])
                for _ in range(self.num_samples):
                    line = self.file.readline()
                    if not line:
                        break
                    sample = orjson.loads(line)
                    yield self._load_sample(sample)
            finally:
                self.file.seek(original_pos)

    def close(self):
        if self.mmap:
            if hasattr(self, "mmap"):
                self.mmap.close()
        else:
            self.file.close()
        if hasattr(self, "_index_tmp") and os.path.exists(self._index_tmp):
            os.remove(self._index_tmp)
        if hasattr(self, "offsets"):
            self.offsets.close()
        if self.bin:
            self.bin.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
