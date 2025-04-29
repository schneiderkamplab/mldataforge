import base64
import bisect
import brotli
import bz2
import gzip
import orjson
import lz4.frame
import lzma
import numpy as np
import os
import snappy
import tempfile
import zstandard as zstd
from pathlib import Path

def decompress_data(data, ext):
    if ext == "zst":
        return zstd.ZstdDecompressor().decompress(data)
    elif ext == "bz2":
        return bz2.decompress(data)
    elif ext == "lzma" or ext == "xz":
        return lzma.decompress(data)
    elif ext == "lz4":
        return lz4.frame.decompress(data)
    elif ext == "snappy":
        return snappy.decompress(data)
    elif ext == "gz":
        return gzip.decompress(data)
    elif ext == "br":
        return brotli.decompress(data)
    else:
        raise ValueError(f"Unsupported compression extension: {ext}")

def compress_data(data, ext):
    if ext is None:
        return data
    if ext == "zst":
        return zstd.ZstdCompressor(level=1).compress(data)
    elif ext == "bz2":
        return bz2.compress(data)
    elif ext == "lzma" or ext == "xz":
        return lzma.compress(data)
    elif ext == "lz4":
        return lz4.frame.compress(data)
    elif ext == "snappy":
        return snappy.compress(data)
    elif ext == "gz":
        return gzip.compress(data)
    elif ext == "br":
        return brotli.compress(data)
    else:
        raise ValueError(f"Unsupported compression extension: {ext}")

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
        """Recursively compress dict fields."""
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
            offsets_bytes = f.read()

        # Compress the offsets
        compressed_index = compress_data(offsets_bytes, self.index_compression)
        encoded_index = base64.a85encode(compressed_index).decode("utf-8")

        index_key = "index" if (self.index_compression is None or self.index_compression == "none") else f"index.{self.index_compression}"

        # Write header
        header_offset = self.current_offset

        header = {
            index_key: encoded_index,
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

        # Write header and final offset
        self.file.write(header_json)
        self.file.write(f"\n{header_offset}\n".encode("utf-8"))
        self.file.close()

        # Clean up temporary offset file
        os.remove(self.offsets_tmp_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If user forgets to close, still call close with minimal required args
        self.close(shard_id=-1)

    def tell(self):
        return self.current_offset

class JinxWriter:

    _SHARD_TEMPLATE = "shard-{shard_id:05d}.jinx"

    def __init__(
        self,
        output_path: str,
        shard_size: int,
        split=None,
        append=False,
        compression="zst",
        index_compression="zst",
        encoding="base85",
        compress_threshold=128,
        compress_ratio=0.67,
    ):
        self.output_path = Path(output_path)
        self.shard_size = shard_size
        self.split = split
        self.append = append
        self.compression = compression
        self.index_compression = index_compression
        self.encoding = encoding
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio
        self.previous_shard_path = None
        self.shard_id = 0
        if self.shard_size is None:
            self.current_path = str(self.output_path)
        else:
            os.makedirs(self.output_path, exist_ok=True)
            while True:
                self.current_path = str(self.output_path / self._SHARD_TEMPLATE.format(shard_id=self.shard_id))
                if not append or not os.path.exists(self.current_path):
                    break
                self.shard_id += 1
        self._open_writer()

    def _open_writer(self):
        self.current_writer = JinxShardWriter(
            path=self.current_path,
            encoding=self.encoding,
            compress_threshold=self.compress_threshold,
            compress_ratio=self.compress_ratio,
            compression=self.compression,
            index_compression=self.index_compression,
        )

    def _new_shard(self):
        next_shard_path = str(self.output_path / self._SHARD_TEMPLATE.format(shard_id=self.shard_id + 1))
        self._close_writer(next_shard_path=next_shard_path)
        self.previous_shard_path = str(self.current_path)
        self.shard_id += 1
        self.current_path = str(next_shard_path)
        self._open_writer()

    def _close_writer(self, next_shard_path=None):
        if self.current_writer:
            self.current_writer.close(
                shard_id=self.shard_id,
                shard_prev=self.previous_shard_path,
                shard_next=next_shard_path,
                split=self.split,
            )
            self.current_writer = None

    def write(self, sample: dict):
        if self.shard_size is not None and self.current_writer.tell() > self.shard_size:
            self._new_shard()
        self.current_writer.write_sample(sample)

    def close(self):
        self._close_writer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def tell(self):
        return self.current_writer.tell() if self.current_writer else 0

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

        if "index" in self.header:
            index_value = self.header["index"]
            if isinstance(index_value, list):
                # Uncompressed, direct offsets
                self.offsets = np.array(index_value, dtype=np.uint64)
            elif isinstance(index_value, str):
                # It's a base85-encoded string, assume raw uint64 array
                index_bytes = base64.a85decode(index_value.encode("utf-8"))
                self.offsets = np.frombuffer(index_bytes, dtype=np.uint64)
            else:
                raise ValueError(f"Unsupported type for index: {type(index_value)}")
        else:
            # Look for compressed index with extensions
            index_key = next((k for k in self.header if k.startswith("index.")), None)
            if index_key:
                ext = index_key.split(".", 1)[-1]
                index_bytes = base64.a85decode(self.header[index_key].encode("utf-8"))
                index_decompressed = decompress_data(index_bytes, ext)
                self.offsets = np.frombuffer(index_decompressed, dtype=np.uint64)
            else:
                raise ValueError("Missing index in JINX header.")

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
                decoded = base64.a85decode(value.encode("ascii"))
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
                line = line.decode("utf-8")
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

class JinxDatasetReader:
    def __init__(self, input_paths, split=None):
        if isinstance(input_paths, (str, Path)):
            input_paths = [input_paths]

        self.shard_paths = []
        for input_path in input_paths:
            input_path = Path(input_path)
            if input_path.is_dir():
                self.shard_paths.extend(sorted(input_path.glob("shard-*.jinx")))
            else:
                self.shard_paths.append(input_path)

        self.shards = []
        self.lengths = []
        self.cumulative_lengths = []

        total = 0
        for path in self.shard_paths:
            shard = JinxShardReader(path, split=split)
            self.shards.append(shard)
            self.lengths.append(len(shard))
            total += len(shard)
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index out of range: {idx}")

        shard_idx = self._find_shard(idx)
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_lengths[shard_idx - 1]

        return self.shards[shard_idx][local_idx]

    def _find_shard(self, global_idx):
        return bisect.bisect_right(self.cumulative_lengths, global_idx)

    def __iter__(self):
        for shard in self.shards:
            yield from shard

    def close(self):
        for shard in self.shards:
            shard.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
