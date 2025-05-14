import bisect
from pathlib import Path

from .shard_reader import JinxShardReader

__all__ = ["JinxDatasetReader"]

class JinxDatasetReader:
    def __init__(self, input_paths, split=None, lazy=False, mmap=False, encoding=None):
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
            shard = JinxShardReader(path, split=split, lazy=lazy, mmap=mmap, encoding=encoding)
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
