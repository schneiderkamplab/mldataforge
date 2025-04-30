import base64
import io
import orjson
import numpy as np
import os
import tempfile
from pathlib import Path

from ..compression import decompress_data, decompress_file
from ..encoding import decode_base85_stream_to_file

__all__ = ["JinxShardReader"]

class JinxShardReader:
    def __init__(self, path: str, split=None):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.bin_path = self.path.with_suffix(".bin")
        self._load_header(split=split)
        self.bin = open(self.bin_path, "rb") if self.bin_path.exists() else None

    def _load_header(self, split=None):
        self.file.seek(-64, 2)
        last_part = self.file.read()
        lines = last_part.strip().split(b"\n")
        footer_offset = int(lines[-1].decode("utf-8"))
        self.file.seek(footer_offset)
        header_line = self.file.readline().decode("utf-8")
        self.header = orjson.loads(header_line)
        self.num_samples = self.header["num_samples"]
        index_key = next((k for k in self.header if k.startswith("index.")), None)
        if index_key is None:
            raise ValueError("Missing index in JINX header.")
        extensions = index_key.split(".")[1:]
        if extensions[-1] == "bin":
            location = self.header[index_key]
            offset, length = location["offset"], location["length"]
            self.offsets = np.memmap(self.bin_path, dtype=np.uint64, mode="r", offset=offset, shape=(length // 8))
        else:
            self._index_tmp = tempfile.NamedTemporaryFile(delete=False).name
            decode_base85_stream_to_file(self.header[index_key], self._index_tmp)
            for i, ext in reversed(list(enumerate(extensions))):
                if ext == "npy":
                    assert i == 0, "Numpy arrays should be the first extension"
                    try:
                        self.offsets = np.memmap(self._index_tmp, dtype=np.uint64, mode="r")
                    except Exception as e:
                        raise ValueError(f"Failed to load .npy array for key 'index': {e}")
                elif ext in {"zst", "bz2", "lz4", "lzma", "snappy", "xz", "gz", "br"}:
                    tmp_path = tempfile.NamedTemporaryFile(delete=False).name
                    decompress_file(self._index_tmp, tmp_path, ext)
                    os.remove(self._index_tmp)
                    self._index_tmp = tmp_path
                else:
                    raise ValueError(f"Unsupported compression type: {ext}")
        if split is not None and "split" in self.header and split != self.header["split"]:
            self.num_samples = 0
        self.encoding = self.header.get("encoding", "base85")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not (0 <= idx < self.num_samples):
            raise IndexError(f"Sample index out of range: {idx}")
        offset = self.offsets[idx]
        self.file.seek(offset)
        line = self.file.readline().decode("utf-8")
        sample = orjson.loads(line)
        return self._decompress_sample(sample)

    def _maybe_decompress_field(self, key, value):
        if "." not in key:
            return key, value
        parts = key.split(".")
        base_key = parts[0]
        extensions = parts[1:]
        assert extensions, "Extensions should not be empty"
        if extensions[-1] == "bin":
            location = value
            offset, length = location["offset"], location["length"]
            self.bin.seek(offset)
            decoded = self.bin.read(length)
            extensions.pop()
        else:
            try:
                decoded = base64.a85decode(value.encode("ascii"), foldspaces=True)
            except Exception as e:
                raise ValueError(f"Failed to decode base85 for key '{key}': {e}")
        for i, ext in reversed(list(enumerate(extensions))):
            if ext == "npy":
                try:
                    return base_key, np.load(io.BytesIO(decoded), allow_pickle=False)
                except Exception as e:
                    raise ValueError(f"Failed to load .npy array for key '{key}': {e}")
            elif ext == "raw":
                try:
                    return base_key, decoded
                except Exception as e:
                    raise ValueError(f"Failed to load binary for key '{key}': {e}")
            elif ext == "np":
                try:
                    dtype_str, data = decoded.split(b"\x00", 1)
                    return base_key, np.frombuffer(data, dtype=dtype_str.decode("ascii"))[0]
                except Exception as e:
                    raise ValueError(f"Failed to load .np array for key '{key}': {e}")
            elif ext in {"zst", "bz2", "lz4", "lzma", "snappy", "xz", "gz", "br"}:
                try:
                    decoded = decompress_data(decoded, ext)
                except Exception as e:
                    raise ValueError(f"Failed to decompress with '{ext}' for key '{key}': {e}")
            else:
                break
        try:
            return base_key, orjson.loads(decoded)
        except Exception as e:
            raise ValueError(f"Failed to decode JSON for key '{key}': {e}")

    def _decompress_sample(self, value):
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                new_key, new_value = self._maybe_decompress_field(k, self._decompress_sample(v))
                result[new_key] = new_value
            return result
        elif isinstance(value, list):
            return [self._decompress_sample(v) for v in value]
        else:
            return value

    def __iter__(self):
        original_pos = self.file.tell()
        try:
            self.file.seek(self.offsets[0])
            for _ in range(self.num_samples):
                line = self.file.readline()
                if not line:
                    break
                sample = orjson.loads(line)
                yield self._decompress_sample(sample)
        finally:
            self.file.seek(original_pos)

    def close(self):
        self.file.close()
        if hasattr(self, "_index_tmp") and os.path.exists(self._index_tmp):
            os.remove(self._index_tmp)
        if hasattr(self, "offerts"):
            self.offsets.close()
        if self.bin:
            self.bin.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
