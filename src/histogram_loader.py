"""
Histogram loader that handles pickles saved with older matplotlib versions.
"""

import pickle
import numpy as np


class _DummyMeta(type):
    """Metaclass that catches class-level attribute access (e.g. Dummy.projecting)."""
    def __getattr__(cls, name):
        return cls()


class _Dummy(metaclass=_DummyMeta):
    """No-op stand-in for any matplotlib class encountered during unpickling."""
    def __new__(cls, *a, **kw): return object.__new__(cls)
    def __init__(self, *a, **kw): pass
    def __setstate__(self, s): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, name): return self
    def __iter__(self): return iter([])
    def __len__(self): return 0
    def __bool__(self): return True
    def __float__(self): return 0.0
    def __int__(self): return 0
    def __index__(self): return 0


class _SkipMatplotlibUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'matplotlib' in module:
            return _Dummy
        return super().find_class(module, name)


def load_histogram(path: str) -> dict:
    """
    Load a simulation histogram pickle file.

    The pickles were saved with an older matplotlib and contain a dict with:
      - 'counts' : np.ndarray  — normalised probability per bin (sums to ~1)
      - 'bins'   : np.ndarray  — bin edges (length = len(counts) + 1)
      - 'bars'   : matplotlib BarContainer (skipped/ignored)

    Returns
    -------
    dict with keys 'counts' and 'bins' as numpy arrays.
    """
    with open(path, 'rb') as f:
        raw = _SkipMatplotlibUnpickler(f).load()

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {path}, got {type(raw).__name__}")

    counts = np.asarray(raw['counts'], dtype=float)
    bins = np.asarray(raw['bins'], dtype=float)

    total = counts.sum()
    if total > 0 and abs(total - 1.0) > 1e-6:
        counts = counts / total

    return {'counts': counts, 'bins': bins}
