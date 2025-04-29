import base64
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
        self._load_header(split=split)

    def _load_header(self, split=None):
        self.file.seek(-64, 2)
        last_part = self.file.read()
        lines = last_part.strip().split(b"\n")
        footer_offset = int(lines[-1].decode("utf-8"))
        self.file.seek(footer_offset)
        header_line = self.file.readline().decode("utf-8")
        self.header = orjson.loads(header_line)
        index_key = "index"
        ext = None
        if "index" not in self.header:
            index_key = next((k for k in self.header if k.startswith("index.")), None)
            if index_key is None:
                raise ValueError("Missing index in JINX header.")
            ext = index_key.split(".", 1)[-1]
        index_value = self.header[index_key]
        if isinstance(index_value, list):
            self.offsets = np.array(index_value, dtype=np.uint64)
            self._index_tempfile = None
        elif isinstance(index_value, str):
            tmp_base85_path = tempfile.NamedTemporaryFile(delete=False).name
            decode_base85_stream_to_file(index_value, tmp_base85_path)
            if ext is not None:
                tmp_decompressed_path = tempfile.NamedTemporaryFile(delete=False).name
                decompress_file(tmp_base85_path, tmp_decompressed_path, ext)
                os.remove(tmp_base85_path)
                tmp_path = tmp_decompressed_path
            else:
                tmp_path = tmp_base85_path
            self._index_tempfile = tmp_path
            if os.path.getsize(tmp_path) == 0:
                raise ValueError("Decoded index file is empty — possibly corrupted or invalid base85/compression.")
            self.offsets = np.memmap(tmp_path, dtype=np.uint64, mode="r")
        else:
            raise ValueError(f"Unsupported type for index: {type(index_value)}")
        self.num_samples = self.header["num_samples"]
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
        if "." in key:
            base_key, ext = key.rsplit(".", 1)
            if ext in {"zst", "bz2", "lz4", "lzma", "snappy", "xz", "gz", "br"}:
                decoded = base64.a85decode(value.encode("ascii"), foldspaces=True)
                decompressed_bytes = decompress_data(decoded, ext)
                try:
                    return base_key, orjson.loads(decompressed_bytes)
                except Exception:
                    return base_key, decompressed_bytes.decode("utf-8")
        return key, value

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
