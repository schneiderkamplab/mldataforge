import base64
import io
import numpy as np
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
        compress_threshold=2**7,
        compress_ratio=0.67,
        compression="zst",
        index_compression="zst",
        binary_threshold=None,
    ):
        self.path = Path(path)
        self.encoding = encoding
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio
        self.compression = compression
        self.index_compression = index_compression
        self.binary_threshold = binary_threshold
        self.file = self.path.open("wb")
        self.current_offset = 0
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.offsets_tmp_path = tmp_file.name
        tmp_file.close()
        self.offsets_file = open(self.offsets_tmp_path, "wb")
        self.num_offsets = 0
        self.bin = None

    def _maybe_compress(self, value, header=False):
        always = isinstance(value, (bytes, np.ndarray, np.generic))
        if self.compression is None and not always and not (isinstance(value, str) and self.binary_threshold and len(value) >= self.binary_threshold):
            return value, None
        ext = None
        if isinstance(value, np.ndarray):
            if header:
                serialized = value.tobytes()
            else:
                buf = io.BytesIO()
                np.save(buf, value, allow_pickle=False)
                buf.seek(0)
                serialized = buf.read()
            ext = "npy"
        elif isinstance(value, bytes):
            serialized = value
            ext = "raw"
        elif isinstance(value, np.generic):
            serialized = str(value.dtype).encode("ascii")+b"\x00"+np.array(value).tobytes()
            ext = "np"
        else:
            serialized = orjson.dumps(value)
        if not always and len(serialized) < self.compress_threshold:
            return value, ext
        compressed = compress_data(serialized, self.compression)
        if len(compressed) <= self.compress_ratio * len(serialized):
            extensions = [x for x in (ext, self.compression) if x]
            if self.binary_threshold and len(compressed) > self.binary_threshold:
                if not self.bin:
                    self.bin = open(self.path.with_suffix(".bin"), "wb")
                offset = self.bin.tell()
                if header:
                    compressed = serialized
                self.bin.write(compressed)
                extensions.append("bin")
                encoded = {"offset": offset, "length": len(compressed)}
                ext = ".".join(extensions)
            elif self.encoding == "base85":
                encoded = base64.a85encode(compressed).decode("utf-8")
            elif self.encoding == "base64":
                encoded = base64.b64encode(compressed).decode("utf-8")
            else:
                raise ValueError(f"Unsupported encoding: {self.encoding}")
            return encoded, ".".join(extensions)
        elif always or (self.binary_threshold and len(serialized) > self.binary_threshold):
            extensions = [x for x in (ext,) if x]
            if self.binary_threshold and len(serialized) > self.binary_threshold:
                if not self.bin:
                    self.bin = open(self.path.with_suffix(".bin"), "wb")
                offset = self.bin.tell()
                self.bin.write(serialized)
                extensions.append("bin")
                encoded = {"offset": offset, "length": len(serialized)}
                ext = ".".join(extensions)
            elif self.encoding == "base85":
                encoded = base64.a85encode(serialized).decode("utf-8")
            elif self.encoding == "base64":
                encoded = base64.b64encode(serialized).decode("utf-8")
            else:
                raise ValueError(f"Unsupported encoding: {self.encoding}")
            return encoded, ext
        return value, ext

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
            offsets = np.fromfile(f, dtype=np.uint64)
        index, ext = self._maybe_compress(offsets, header=True)
        index_key = f"index.{ext}"
        header_offset = self.current_offset
        header = {
            index_key: index,
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
        if self.bin:
            self.bin.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(shard_id=-1)

    def tell(self):
        return self.current_offset
