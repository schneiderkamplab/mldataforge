import base64
import io
import orjson
import numpy as np
import os
import tempfile
from pathlib import Path

from ..compression import decompress_data, decompress_file
from ..encoding import decode_a85_stream_to_file

__all__ = ["JinxShardReader"]

class JinxShardReader:
    def __init__(self, path: str, split=None):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.bin_path = self.path.with_suffix(".bin")
        self._load_footer(split=split)
        self.bin = open(self.bin_path, "rb") if self.bin_path.exists() else None

    def _load_footer(self, split=None):
        self.file.seek(-64, os.SEEK_END)
        last_part = self.file.read()
        lines = last_part.strip().split(b"\n")
        footer_offset = int(lines[-1].decode("utf-8"))
        self.file.seek(footer_offset)
        header_line = self.file.readline().decode("utf-8")
        self.header = orjson.loads(header_line)
        self.num_samples = self.header["num_samples"]
        if split is not None and self.header.get("split") != split:
            self.num_samples = 0
        index_key = next((k for k in self.header if k.startswith("index.")), None)
        if index_key is None:
            raise ValueError("Missing index in JINX header.")
        index_data = self.header[index_key]
        extensions = index_key.split(".")[1:]
        self.offsets = self._mmap_index(index_data, extensions)

    def _mmap_index(self, data, extensions):
        if extensions[-1] == "bin":
            location = data
            offset, length = location["offset"], location["length"]
            return np.memmap(self.bin_path, dtype=np.uint64, mode="r", offset=offset, shape=(length // 8))
        tmp_path = tempfile.NamedTemporaryFile(delete=False).name
        decode_a85_stream_to_file(data, tmp_path)
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
        if not (0 <= idx < self.num_samples):
            raise IndexError(f"Sample index out of range: {idx}")
        offset = self.offsets[idx]
        self.file.seek(offset)
        line = self.file.readline().decode("utf-8")
        sample = orjson.loads(line)
        return self._load_sample(sample)

    def _load_value(self, key, value):
        if "." not in key:
            return key, value
        parts = key.split(".")
        base_key, extensions = parts[0], parts[1:]
        scalar = self._load_scalar(key, value, extensions)
        if scalar is not None:
            return base_key, scalar
        decoded, remaining_extensions = self._load_bytes(key, value, extensions)
        decoded_value = self._unserialize(key, decoded, remaining_extensions, original_value=value)
        return base_key, decoded_value

    def _load_scalar(self, key, value, extensions):
        if len(extensions) == 1 and extensions[0] in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"}:
            try:
                return np.dtype(extensions[0]).type(value)
            except Exception as e:
                raise ValueError(f"Failed to cast value to {extensions[0]} for key '{key}': {e}")
        return None

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
        exts = list(extensions)  # make a copy we can modify
        if exts[-1] == "bin":
            if not isinstance(value, dict) or "offset" not in value:
                raise ValueError(f"Expected offset dict for '.bin' extension in key '{key}'")
            offset, length = value["offset"], value["length"]
            self.bin.seek(offset)
            decoded = self.bin.read(length)
            exts.pop()  # remove "bin" from extension list
        elif isinstance(value, str):
            try:
                decoded = base64.a85decode(value.encode("ascii"), foldspaces=True)
            except Exception as e:
                raise ValueError(f"Failed to decode base85 for key '{key}': {e}")
        else:
            # No binary encoding/decoding — native value
            return value, []
        return decoded, exts

    def _load_sample(self, value):
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                new_key, new_value = self._load_value(k, self._load_sample(v))
                result[new_key] = new_value
            return result
        elif isinstance(value, list):
            return [self._load_sample(v) for v in value]
        else:
            return value

    def __iter__(self):
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
