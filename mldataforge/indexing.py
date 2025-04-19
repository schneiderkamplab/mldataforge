import numpy as np

__all__ = ['IndexedDatasetView', 'shuffle_permutation']

class IndexedDatasetView:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)  # ensure repeatable accessx

    def __iter__(self):
        for idx in self.indices:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.indices)

def shuffle_permutation(n, seed=int):
    rng = np.random.default_rng(seed)
    return rng.permutation(n)

def reverse_permutation(indices):
    n = len(indices)
    reverse_indices = np.empty(n, dtype=int)
    reverse_indices[indices] = np.arange(n)
    return reverse_indices
