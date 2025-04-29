import base64
import orjson
import os
import tempfile
from pathlib import Path

from ..compression import compress_data

__all__ = ["JinxShardWriter"]

class JinxShardWriter:
    def __init__(
        self,
        path: str,
        encoding="base85",
        compress_threshold=128,
        compress_ratio=0.67,
        compression="zst",
        index_compression="zst"
    ):
        self.path = Path(path)
        self.encoding = encoding
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio
        self.compression = compression
        self.index_compression = index_compression
        self.file = self.path.open("wb")
        self.current_offset = 0
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.offsets_tmp_path = tmp_file.name
        tmp_file.close()
        self.offsets_file = open(self.offsets_tmp_path, "wb")
        self.num_offsets = 0

    def _maybe_compress(self, value):
        if self.compression is None:
            return value, None
        serialized = orjson.dumps(value)
        if len(serialized) < self.compress_threshold:
            return value, None
        compressed = compress_data(serialized, self.compression)
        if len(compressed) <= self.compress_ratio * len(serialized):
            if self.encoding == "base85":
                encoded = base64.a85encode(compressed).decode("utf-8")
            elif self.encoding == "base64":
                encoded = base64.b64encode(compressed).decode("utf-8")
            else:
                raise ValueError(f"Unsupported encoding: {self.encoding}")
            return encoded, self.compression
        return value, None

    def _maybe_compress_recursive(self, value):
        if isinstance(value, dict):
            new_dict = {}
            for k, v in value.items():
                compressed_value, compression_type = self._maybe_compress_recursive(v)
                if compression_type:
                    k = f"{k}.{compression_type}"
                new_dict[k] = compressed_value
            return new_dict, None
        elif isinstance(value, list):
            return [self._maybe_compress_recursive(v)[0] for v in value], None
        else:
            compressed_value, compression_type = self._maybe_compress(value)
            return compressed_value, compression_type

    def write_sample(self, sample: dict):
        compressed_sample, _ = self._maybe_compress_recursive(sample)
        json_line = orjson.dumps(compressed_sample)
        self.offsets_file.write(self.current_offset.to_bytes(8, byteorder='little'))
        self.file.write(json_line)
        self.file.write(b"\n")
        self.num_offsets += 1
        self.current_offset += len(json_line)+1

    def close(self, shard_id: int, shard_prev: str = None, shard_next: str = None,
              split: str = None, dataset_name: str = None, hash_value: str = None):
        self.offsets_file.close()
        with open(self.offsets_tmp_path, "rb") as f:
            raw_index_bytes = f.read()
        index_key = "index"
        if self.index_compression:
            compressed_index = compress_data(raw_index_bytes, self.index_compression)
            encoded = base64.a85encode(compressed_index).decode("utf-8")
            index_key = f"index.{self.index_compression}"
        else:
            encoded = base64.a85encode(raw_index_bytes).decode("utf-8")
        header_offset = self.current_offset
        header = {
            index_key: encoded,
            "encoding": self.encoding,
            "num_samples": self.num_offsets,
            "shard_id": shard_id,
            "version": "1.0",
        }
        if shard_prev:
            header["shard_prev"] = shard_prev
        if shard_next:
            header["shard_next"] = shard_next
        if split:
            header["split"] = split
        if dataset_name:
            header["dataset_name"] = dataset_name
        if hash_value:
            header["hash"] = hash_value
        header_json = orjson.dumps(header)
        self.file.write(header_json)
        self.file.write(f"\n{header_offset}\n".encode("utf-8"))
        self.file.close()
        os.remove(self.offsets_tmp_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(shard_id=-1)

    def tell(self):
        return self.current_offset
