import base64
import bisect
import brotli
import bz2
import gzip
import json
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
    if ext is None or ext == "none":
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
    def __init__(self, path: str, encoding="base85", compress_threshold=128, compress_ratio=0.67, compression="zst", index_compression="zst"):
        self.path = Path(path)
        self.encoding = encoding
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio
        self.compression = compression
        self.index_compression = index_compression

        self.file = self.path.open("w", encoding="utf-8")

        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.offsets_tmp_path = tmp_file.name
        tmp_file.close()
        self.offsets_file = open(self.offsets_tmp_path, "w+b")
        self.num_offsets = 0

    def _maybe_compress(self, value):
        if isinstance(value, str):
            data = value.encode("utf-8")
        else:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")

        if len(data) < self.compress_threshold:
            return value, None

        compressed = compress_data(data, self.compression)
        if len(compressed) <= self.compress_ratio * len(data):
            if self.encoding == "base85":
                encoded = base64.a85encode(compressed).decode("utf-8")
            else:
                raise ValueError(f"Unsupported encoding: {self.encoding}")
            return encoded, self.compression if self.compression else "none"

        return value, None

    def write_sample(self, sample: dict):
        offset = self.file.tell()

        self.offsets_file.seek(self.num_offsets * 8)
        self.offsets_file.write(offset.to_bytes(8, byteorder='little'))
        self.num_offsets += 1

        new_sample = {}
        for k, v in sample.items():
            compressed_value, compression_type = self._maybe_compress(v)
            if compression_type and compression_type != "none":
                k = k + "." + compression_type
            new_sample[k] = compressed_value

        json_line = json.dumps(new_sample, ensure_ascii=False)
        self.file.write(json_line + "\n")

    def close(self, shard_id: int, shard_prev: str = None, shard_next: str = None, split: str = None, dataset_name: str = None, hash_value: str = None):
        header_offset = self.file.tell()

        self.offsets_file.flush()
        self.offsets_file.close()

        with open(self.offsets_tmp_path, "rb") as f:
            offsets_bytes = f.read()

        compressed_index = compress_data(offsets_bytes, self.index_compression)
        encoded_index = base64.a85encode(compressed_index).decode("utf-8")

        index_key = "index" if (self.index_compression is None or self.index_compression == "none") else f"index.{self.index_compression}"

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

        header_json = json.dumps(header, ensure_ascii=False)
        self.file.write(header_json + "\n")
        self.file.write(str(header_offset) + "\n")

        self.file.close()
        os.remove(self.offsets_tmp_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def tell(self):
        return self.file.tell()

class JinxWriter:
    def __init__(self, output_path: str, shard_size: int, compression="zst", index_compression="zst", encoding="base85", compress_threshold=128, compress_ratio=0.67):
        self.output_path = Path(output_path)
        self.shard_size = shard_size
        self.compression = compression
        self.index_compression = index_compression
        self.encoding = encoding
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio

        self.shard_id = 0
        self.current_writer = None
        self.current_path = self.output_path
        self.current_shard_bytes = 0
        self.previous_shard_name = None
        self.sharded = False

        self._open_new_shard()

    def _open_new_shard(self):
        if self.current_writer:
            self.current_writer.close(
                shard_id=self.shard_id,
                shard_prev=self.previous_shard_name,
                shard_next=self._next_shard_name(self.shard_id + 1),
            )
            self.previous_shard_name = self.current_path.name
            self.shard_id += 1

        if self.shard_id > 0 and not self.sharded:
            # Transition to a directory of shards
            os.rename(self.output_path, self.output_path.with_suffix('.tmp'))
            dir_path = self.output_path
            dir_path.mkdir(parents=True, exist_ok=True)
            first_shard = dir_path / f"shard-00000.jinx"
            os.rename(self.output_path.with_suffix('.tmp'), first_shard)
            self.output_path = dir_path
            self.sharded = True

        if self.sharded:
            self.current_path = self.output_path / f"shard-{self.shard_id:05d}.jinx"
        else:
            self.current_path = self.output_path

        self.current_writer = JinxShardWriter(
            path=self.current_path,
            encoding=self.encoding,
            compress_threshold=self.compress_threshold,
            compress_ratio=self.compress_ratio,
            compression=self.compression,
            index_compression=self.index_compression,
        )
        self.current_shard_bytes = 0

    def _next_shard_name(self, shard_id):
        if self.sharded:
            return f"shard-{shard_id:05d}.jinx"
        else:
            return self.output_path.name

    def write(self, sample: dict):
        sample_json = json.dumps(sample, ensure_ascii=False)
        sample_size = len(sample_json.encode("utf-8"))

        if self.shard_size and self.current_shard_bytes + sample_size > self.shard_size and self.current_shard_bytes > 0:
            self._open_new_shard()

        self.current_writer.write_sample(sample)
        self.current_shard_bytes += sample_size

    def close(self):
        if self.current_writer:
            self.current_writer.close(
                shard_id=self.shard_id,
                shard_prev=self.previous_shard_name,
                shard_next=None,
            )
            self.current_writer = None

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

        header_line = self.file.readline().decode("utf-8").strip()
        self.header = json.loads(header_line)

        index_key = next((k for k in self.header if k.startswith("index.")), None)
        if index_key:
            ext = index_key.split(".", 1)[-1]
            index_bytes = base64.a85decode(self.header[index_key].encode("utf-8"))
            index_decompressed = decompress_data(index_bytes, ext)
            self.offsets = np.frombuffer(index_decompressed, dtype=np.uint64)
        else:
            raise ValueError("Missing compressed index in JINX header.")

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
        line = self.file.readline().decode("utf-8").strip()
        sample = json.loads(line)

        return self._decompress_sample(sample)

    def _decompress_sample(self, sample):
        decompressed = {}
        for key, value in sample.items():
            if "." in key:
                base_key, ext = key.rsplit(".", 1)
                if ext in {"zst", "bz2", "lz4", "lzma", "snappy", "xz", "gz", "br"}:
                    data = base64.a85decode(value.encode("utf-8"))
                    decompressed_data = decompress_data(data, ext)
                    try:
                        decompressed[base_key] = json.loads(decompressed_data.decode("utf-8"))
                    except json.JSONDecodeError:
                        decompressed[base_key] = decompressed_data.decode("utf-8")
                else:
                    decompressed[base_key] = value
            else:
                decompressed[key] = value

        return decompressed

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

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
