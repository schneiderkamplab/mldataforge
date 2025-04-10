import gzip
import json
from mltiming import timing
import numpy as np
import os
from streaming.base.format.mds.encodings import mds_decode
from typing import Any, Optional, Generator

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
        if compression is None:
            _open = open
        elif compression == 'gz':
            _open = gzip.open
        else:
            raise ValueError(f'Unsupported compression type: {compression}. Supported types: None, gzip.')
        self.fp = _open(filename, "rb")
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
