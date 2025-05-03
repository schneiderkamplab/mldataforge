import numpy as np

__all__ = ['IndexedDatasetView', 'identity_permutation', 'process_indices', 'reverse_permutation', 'shuffle_permutation', 'sort_permutation']

class IndexedDatasetView:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        length = len(dataset)
        self.indices = [i for i in indices if 0 <= i < length]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.dataset[self.indices[index]]
        elif isinstance(index, slice):
            return [self.dataset[i] for i in self.indices[index]]
        else:
            raise TypeError("Index must be an int or a slice")

    def __iter__(self):
        for idx in self.indices:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.indices)

def identity_permutation(n):
    return np.arange(n, dtype=np.uint64)

def process_indices(indices, every=None, offset=None, number=None):
    if every is not None:
        indices = indices[::every]
    if offset is not None:
        indices = indices[offset:]
    if number is not None:
        indices = indices[:number]
    return indices

def shuffle_permutation(n, seed=int):
    rng = np.random.default_rng(seed)
    return rng.permutation(n).astype(np.uint64)

def sort_permutation(dataset, sort_key):
    if isinstance(sort_key, str):
        global_context = {}
        exec(sort_key, global_context)
        if 'key' not in global_context or not callable(global_context['key']):
            raise ValueError("code must define a callable named 'key'")
        sort_key = global_context['key']
    indices = np.argsort([sort_key(item) for item in dataset], kind='stable')
    return indices.astype(np.uint64)

def reverse_permutation(indices):
    n = len(indices)
    reverse_indices = np.empty(n, dtype=np.uint64)
    reverse_indices[indices] = np.arange(n, dtype=np.uint64)
    return reverse_indices
