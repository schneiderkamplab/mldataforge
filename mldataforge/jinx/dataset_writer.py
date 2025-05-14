import os
from pathlib import Path

from .shard_writer import JinxShardWriter

__all__ = ["JinxDatasetWriter"]

class JinxDatasetWriter:

    _SHARD_TEMPLATE = "shard-{shard_id:05d}.jinx"

    def __init__(
        self,
        output_path: str,
        shard_size: int,
        split=None,
        append=False,
        compression="zst",
        index_compression="zst",
        compress_threshold=128,
        compress_ratio=0.67,
        encoding="a85",
        binary_threshold=None,
        ext_sep=".",
    ):
        self.output_path = Path(output_path)
        self.shard_size = shard_size
        self.split = split
        self.append = append
        self.compression = compression
        self.index_compression = index_compression
        self.compress_threshold = compress_threshold
        self.compress_ratio = compress_ratio
        self.encoding = encoding
        self.binary_threshold = binary_threshold
        self.ext_sep = ext_sep
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
            compress_threshold=self.compress_threshold,
            compress_ratio=self.compress_ratio,
            compression=self.compression,
            index_compression=self.index_compression,
            encoding=self.encoding,
            binary_threshold=self.binary_threshold,
            ext_sep=self.ext_sep,
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
        if self.shard_size is not None and self.current_writer.tell()+self.current_writer.num_offsets*8 > self.shard_size:
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
