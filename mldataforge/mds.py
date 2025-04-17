import json
import numpy as np
import os
import shutil
from streaming.base.compression import compress, get_compression_extension, is_compression
from streaming.base.format.index import get_index_basename
from streaming.base.format.mds.encodings import mds_decode, mds_encode, is_mds_encoding, get_mds_encodings, get_mds_encoded_size
from streaming.base.hashing import get_hash, is_hash
from streaming.base.util import bytes_to_int
from typing import Any, Optional, Generator, Self, Union

from .utils import open_compression

class MDSBulkReader:
    def __init__(
        self,
        dirnames: list[str],
        split: Optional[str],
    ) -> None:
        self.shards = []
        self.samples = 0
        for dirname in dirnames:
            if split is not None:
                dirname = os.path.join(dirname, split)
            index = json.load(open(os.path.join(dirname, "index.json"), 'rt'))
            for shard in index["shards"]:
                basename = shard['raw_data']['basename'] if shard['zip_data'] is None else shard['zip_data']['basename']
                filename = os.path.join(dirname, basename)
                self.shards.append({
                    "filename": filename,
                    "compression": shard['compression'],
                })
                self.samples += shard['samples']

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        for shard in self.shards:
            with MDSShardReader(**shard) as reader:
                for sample in reader:
                    yield sample

class MDSShardReader:
    def __init__(
        self,
        filename: str,
        compression: Optional[str],
    ) -> None:
        self.fp = open_compression(filename, "rb", compression=compression)
        self.samples = np.frombuffer(self.fp.read(4), np.uint32)[0]
        self.index = np.frombuffer(self.fp.read((1+self.samples)*4), np.uint32)
        info = json.loads(self.fp.read(self.index[0]-self.fp.tell()))
        self.column_encodings = info["column_encodings"]
        self.column_names = info["column_names"]
        self.column_sizes = info["column_sizes"]
        assert self.fp.tell() == self.index[0]

    def decode_sample(self, data: bytes) -> dict[str, Any]:
        sizes = []
        idx = 0
        for key, size in zip(self.column_names, self.column_sizes):
            if size:
                sizes.append(size)
            else:
                size, = np.frombuffer(data[idx:idx + 4], np.uint32)
                sizes.append(size)
                idx += 4
        sample = {}
        for key, encoding, size in zip(self.column_names, self.column_encodings, sizes):
            value = data[idx:idx + size]
            sample[key] = mds_decode(encoding, value)
            idx += size
        return sample

    def get_sample_data(self, idx: int) -> bytes:
        begin, end = self.index[idx:idx+2]
        assert self.fp.tell() == begin
        data = self.fp.read(end - begin)
        assert self.fp.tell() == end
        assert data
        return data

    def get_item(self, idx: int) -> dict[str, Any]:
        data = self.get_sample_data(idx)
        return self.decode_sample(data)

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        for i in range(self.samples):
            yield self.get_item(i)

    def __enter__(self) -> "MDSShardReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.fp.close()


class MDSWriter:

    format = 'mds'
    extra_bytes_per_sample = 4

    def __init__(self,
                 *,
                 columns: dict[str, str],
                 out: Union[str, tuple[str, str]],
                 compression: Optional[str] = None,
                 hashes: Optional[list[str]] = None,
                 size_limit: Optional[Union[int, str]] = 1 << 26,
                 **kwargs: Any) -> None:
        compression = compression or None
        if compression:
            if not is_compression(compression):
                raise ValueError(f'Invalid compression: {compression}.')

        hashes = hashes or []
        if list(hashes) != sorted(hashes):
            raise ValueError('Hashes must be unique and in sorted order.')
        for algo in hashes:
            if not is_hash(algo):
                raise ValueError(f'Invalid hash: {algo}.')

        size_limit_value = None
        if size_limit:
            size_limit_value = bytes_to_int(size_limit)
            if size_limit_value < 0:
                raise ValueError(f'`size_limit` must be greater than zero, instead, ' +
                                 f'found as {size_limit_value}.')
            if size_limit_value >= 2**32:
                raise ValueError(f'`size_limit` must be less than 2**32, instead, ' +
                                 f'found as {size_limit_value}. This is because sample ' +
                                 f'byte offsets are stored with uint32.')

        # Validate keyword arguments
        invalid_kwargs = [
            arg for arg in kwargs.keys()
            if arg not in ('progress_bar', 'exist_ok')
        ]
        if invalid_kwargs:
            raise ValueError(f'Invalid Writer argument(s): {invalid_kwargs} ')

        self.compression = compression
        self.hashes = hashes
        self.size_limit = size_limit_value
        self.new_samples: list[bytes]
        self.new_shard_size: int

        self.shards = []

        # Remove local directory if requested prior to creating writer
        self.local = os.path.expanduser(out)
        if os.path.exists(self.local) and len(os.listdir(self.local)) != 0:
            if kwargs.get('exist_ok', False):
                raise FileExistsError(f'Directory is not empty: {self.local}')
            shutil.rmtree(self.local)
        os.makedirs(self.local, exist_ok=True)
    
        self.columns = columns
        self.column_names = []
        self.column_encodings = []
        self.column_sizes = []
        for name in sorted(columns):
            encoding = columns[name]
            if not is_mds_encoding(encoding):
                raise TypeError(f'MDSWriter passed column `{name}` with encoding `{encoding}` ' +
                                f'is unsupported. Supported encodings are {get_mds_encodings()}')
            size = get_mds_encoded_size(encoding)
            self.column_names.append(name)
            self.column_encodings.append(encoding)
            self.column_sizes.append(size)

        obj = self.get_config()
        text = json.dumps(obj, sort_keys=True)
        self.config_data = text.encode('utf-8')
        self.extra_bytes_per_shard = 4 + 4 + len(self.config_data)
        self._reset_cache()

    def encode_sample(self, sample: dict[str, Any]) -> bytes:
        sizes = []
        data = []
        for key, encoding, size in zip(self.column_names, self.column_encodings,
                                       self.column_sizes):
            value = sample[key]
            datum = mds_encode(encoding, value)
            if size is None:
                size = len(datum)
                sizes.append(size)
            else:
                if size != len(datum):
                    raise KeyError(f'Unexpected data size; was this data typed with the correct ' +
                                   f'encoding ({encoding})?')
            data.append(datum)
        head = np.array(sizes, np.uint32).tobytes()
        body = b''.join(data)
        return head + body

    def encode_joint_shard(self) -> bytes:
        num_samples = np.uint32(len(self.new_samples))
        sizes = list(map(len, self.new_samples))
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)
        offsets += len(num_samples.tobytes()) + len(offsets.tobytes()) + len(self.config_data)
        sample_data = b''.join(self.new_samples)
        return num_samples.tobytes() + offsets.tobytes() + self.config_data + sample_data

    def flush_shard(self) -> None:
        raw_data_basename, zip_data_basename = self._name_next_shard()
        raw_data = self.encode_joint_shard()
        raw_data_info, zip_data_info = self._process_file(raw_data, raw_data_basename,
                                                          zip_data_basename)
        obj = {
            'samples': len(self.new_samples),
            'raw_data': raw_data_info,
            'zip_data': zip_data_info
        }
        obj.update(self.get_config())
        self.shards.append(obj)

    def _reset_cache(self) -> None:
        self.new_samples = []
        self.new_shard_size = self.extra_bytes_per_shard

    def _name_next_shard(self, extension: Optional[str] = None) -> tuple[str, Optional[str]]:
        shard = len(self.shards)
        parts = ['shard', f'{shard:05}', self.format]
        if extension:
            parts.append(extension)
        raw_basename = '.'.join(parts)
        if self.compression:
            ext = get_compression_extension(self.compression)
            parts.append(ext)
            zip_basename = '.'.join(parts)
        else:
            zip_basename = None
        return raw_basename, zip_basename

    def _hash(self, data: bytes, basename: str) -> dict[str, Any]:
        hashes = {}
        for algo in self.hashes:
            hashes[algo] = get_hash(algo, data)
        return {'basename': basename, 'bytes': len(data), 'hashes': hashes}

    def _process_file(self, raw_data: bytes, raw_basename: str,
                      zip_basename: Optional[str]) -> tuple[dict, Optional[dict]]:
        raw_info = self._hash(raw_data, raw_basename)
        if zip_basename:
            zip_data = compress(self.compression, raw_data)
            zip_info = self._hash(zip_data, zip_basename)
            data = zip_data
            basename = zip_basename
        else:
            zip_info = None
            data = raw_data
            basename = raw_basename
        filename = os.path.join(self.local, basename)
        with open(filename, 'wb') as out:
            out.write(data)
        return raw_info, zip_info

    def get_config(self) -> dict[str, Any]:
        return {
            'version': 2,
            'format': self.format,
            'compression': self.compression,
            'hashes': self.hashes,
            'size_limit': self.size_limit,
            'column_names': self.column_names,
            'column_encodings': self.column_encodings,
            'column_sizes': self.column_sizes,
        }

    def write(self, sample: dict[str, Any]) -> None:
        new_sample = self.encode_sample(sample)
        new_sample_size = len(new_sample) + self.extra_bytes_per_sample
        if self.size_limit and self.size_limit < self.new_shard_size + new_sample_size:
            self.flush_shard()
            self._reset_cache()
        self.new_samples.append(new_sample)
        self.new_shard_size += new_sample_size

    def _write_index(self) -> None:
        if self.new_samples:
            raise RuntimeError('Internal error: not all samples have been written.')
        basename = get_index_basename()
        filename = os.path.join(self.local, basename)
        obj = {
            'version': 2,
            'shards': self.shards,
        }
        with open(filename, 'w') as out:
            json.dump(obj, out, sort_keys=True)

    def finish(self) -> None:
        if self.new_samples:
            self.flush_shard()
            self._reset_cache()
        self._write_index()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish()
