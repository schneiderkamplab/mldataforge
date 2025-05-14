import base64
import io
import numpy as np
import orjson
import os
import tempfile
from pathlib import Path

from ..lazy_dict import LazyDict
from ..compression import compress_data

__all__ = ["JinxShardWriter"]

class JinxShardWriter:
    def __init__(
        self,
        path: str,
        compress_threshold=2**6,
        compress_ratio=1.0,
        compression=None,
        index_compression=None,
        encoding="a85",
        binary_threshold=None,
        ext_sep=".",
    ):
        self.path = Path(path)
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio
        self.compression = compression
        self.index_compression = index_compression
        self.encoding = encoding
        self.binary_threshold = binary_threshold
        self.ext_sep = ext_sep
        self.file = self.path.open("wb")
        self.current_offset = 0
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.offsets_tmp_path = tmp_file.name
        tmp_file.close()
        self.offsets_file = open(self.offsets_tmp_path, "wb")
        self.offsets_file.write(b"\x00" * 8)
        self.num_offsets = 0
        self.bin = None

    def _prepare_value(self, value):
        if isinstance(value, (int, float, bool, type(None))):
            return value, None
        if isinstance(value, str):
            short_limit = min(
                self.compress_threshold if self.compression is not None else float("inf"),
                self.binary_threshold if self.binary_threshold is not None else float("inf")
            )
            if len(value) < short_limit:
                return value, None

            # Encode only here, if string is long enough
            serialized, ext = value.encode("utf-8"), "str"
        else:
            serialized, ext = self._serialize(value)
        return self._handle_bytes(serialized, ext)


    def _serialize(self, value):
        if isinstance(value, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            return buf.getvalue(), "npy"

        elif isinstance(value, bytes):
            return value, "raw"

        elif isinstance(value, np.generic):
            return value.item(), str(value.dtype)

        else:
            try:
                return orjson.dumps(value), None
            except Exception as e:
                raise ValueError(f"Failed to serialize value {value}: {e}")


    def _handle_bytes(self, data, ext):
        # Scalars from np.generic are already native
        if isinstance(data, (int, float, bool, str, type(None))):
            return data, ext

        # Try compression if enabled, large enough, and not going to sidecar
        if self.compression is not None and len(data) >= self.compress_threshold:
            compressed = compress_data(data, self.compression)
            if len(compressed) <= self.compress_ratio * len(data):
                return self._store_data(compressed, ext, compressed=True)

        return self._store_data(data, ext, compressed=False)


    def _store_data(self, data, ext, compressed):
        extensions = [ext, self.compression] if compressed else [ext]
        extensions = [e for e in extensions if e]

        # Sidecar: store large binary data in .bin file
        if self.binary_threshold is not None and len(data) > self.binary_threshold:
            if not self.bin:
                self.bin = open(self.path.with_suffix(".binx"), "wb")
            offset = self.bin.tell()
            self.bin.write(data)
            extensions.append("bin")
            return {"offset": offset, "length": len(data)}, self.ext_sep.join(extensions)

        if extensions == ["str"]:
            # Handle string separately to avoid double-encoding
            return data.decode("utf-8"), None

        # Encode bytes into baseXX for JSON embedding
        if isinstance(data, bytes):
            return self._encode_bytes(data), self.ext_sep.join(extensions)

        return data, self.ext_sep.join(extensions) if extensions else None


    def _encode_bytes(self, data):
        if self.encoding == "a85":
            raw = base64.a85encode(data)
        elif self.encoding == "b64":
            raw = base64.b64encode(data)
        elif self.encoding == "hex":
            raw = data.hex().encode("utf-8")
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")
        return raw.decode("utf-8")

    def _prepare_sample(self, value):
        if isinstance(value, dict):
            new_dict = {}
            for key, val in value.items():
                compressed_val, ext = self._prepare_sample(val)
                if ext:
                    key = f"{key}.{ext}"
                new_dict[key] = compressed_val
            return new_dict, None

        elif isinstance(value, LazyDict):
            new_dict = {}
            for key, val in value.raw_items():
                if value._key_fn(key) == key:
                    compressed_val, ext = self._prepare_sample(val)
                    if ext:
                        key = f"{key}.{ext}"
                    new_dict[key] = compressed_val
                    continue
                if key.endswith(".bin"):
                    offset = val["offset"]
                    length = val["length"]
                    other = value.context
                    if (
                        self.compression != other.header["compression"]
                        or not self.binary_threshold
                        or length < self.binary_threshold
                        or self.compress_ratio != other.header["compress_ratio"]
                        or (self.compression is not None and length < self.compress_threshold)
                    ):
                        # fallback
                        val = other._lazy_load_value(key, val)
                        key = value._key_fn(key)
                        compressed_val, ext = self._prepare_sample(val)
                        if ext:
                            key = f"{key}.{ext}"
                        new_dict[key] = compressed_val
                        continue
                    if not other.bin:
                        other.bin = open(other.path.with_suffix(".binx"), "rb")
                    other.bin.seek(offset)
                    data = other.bin.read(length)
                    if not self.bin:
                        self.bin = open(self.path.with_suffix(".binx"), "wb")
                    offset = self.bin.tell()
                    self.bin.write(data)
                    new_dict[key] = {"offset": offset, "length": length}
                    continue
                new_dict[key] = value.eagerize(val)
            return new_dict, None

        elif isinstance(value, list):
            compressed_list = []
            for item in value:
                compressed_item, _ = self._prepare_sample(item)
                compressed_list.append(compressed_item)
            return compressed_list, None

        else:
            return self._prepare_value(value)

    def _prepare_index(self, array: np.ndarray):
        assert isinstance(array, np.ndarray) and array.dtype == np.uint64, "Index must be a NumPy array with dtype=uint64"
        data = array.tobytes()
        if self.binary_threshold is not None and len(data) > self.binary_threshold:
            if not self.bin:
                self.bin = open(self.path.with_suffix(".binx"), "wb")
            offset = self.bin.tell()
            self.bin.write(data)
            return {"offset": offset, "length": len(data)}, "npy.bin"
        if (
            self.index_compression is not None
            and len(data) >= self.compress_threshold
        ):
            compressed = compress_data(data, self.index_compression)
            if len(compressed) <= self.compress_ratio * len(data):
                data = compressed
                return self._encode_bytes(data), f"npy.{self.index_compression}"
        return self._encode_bytes(data), "npy"


    def write_sample(self, sample: dict):
        prepared_sample, _ = self._prepare_sample(sample)
        json_line = orjson.dumps(prepared_sample)
        self.file.write(json_line)
        self.file.write(b"\n")
        self.num_offsets += 1
        self.current_offset += len(json_line)+1
        self.offsets_file.write(self.current_offset.to_bytes(8, byteorder='little'))

    def close(self, shard_id: int, shard_prev: str = None, shard_next: str = None,
              split: str = None, dataset_name: str = None, hash_value: str = None):
        self.offsets_file.close()
        with open(self.offsets_tmp_path, "rb") as f:
            offsets = np.fromfile(f, dtype=np.uint64)
        index, ext = self._prepare_index(offsets)
        index_key = f"index.{ext}"
        header_offset = self.current_offset
        header = {
            index_key: index,
            "num_samples": self.num_offsets,
            "shard_id": shard_id,
            "version": "1.0",
            "encoding": self.encoding,
            "binary_threshold": self.binary_threshold,
            "compression": self.compression,
            "index_compression": self.index_compression,
            "compress_threshold": self.compress_threshold,
            "compress_ratio": self.compress_ratio,
            "ext_sep": self.ext_sep,
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
