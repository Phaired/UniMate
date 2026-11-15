"""Convert a run's ``dataset_stats.npy`` (pickled dict) to UMTF.

    python tools/python/pipeline_convert_stats.py \
        [models/unimate/unimate_uniml3d_f60_v3/dataset_stats.npy] \
        [models/converted/unimate_uniml3d_f60_v3_stats.umtf]

Which pool does sampling use?  ``MotionDataset.resolve_stats_key`` (upstream
unimate/dataset/mixture/dataset.py): the asset's stats dataset if the file has
it, else ``STATS_FALLBACK`` (general -> objaverse), else any key when all
entries are identical (one global pool). The released f60_v3 file holds one
global pool ('<all>') broadcast to truebones/mixamo/objaverse, so every asset
(the examples are rig_preprocess bundles with stats_dataset='general' ->
objaverse) gets the same numbers.

Every pool is written as ``<key>/mean_root`` ... (f32) plus an exact float64
copy ``<key>/mean_root@f64`` stored as raw little-endian bytes (u8), so C++
can normalise in double exactly like numpy. meta records the keys, the
fallback table and the pool tags.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from umtf import write_umtf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ARRAYS = ('mean_root', 'std_root', 'mean_local', 'std_local')
STATS_FALLBACK = {'general': 'objaverse'}   # upstream dataset.py


def convert(src, dst):
    stats = np.load(src, allow_pickle=True).item()
    tensors, pools = {}, {}
    for key, entry in stats.items():
        for a in ARRAYS:
            arr = np.asarray(entry[a], dtype=np.float64)
            tensors[f'{key}/{a}'] = arr.astype(np.float32)
            tensors[f'{key}/{a}@f64'] = np.frombuffer(arr.astype('<f8').tobytes(), dtype=np.uint8)
        pools[key] = {'pool': str(entry.get('pool')),
                      'balanced_stats': bool(entry.get('balanced_stats', False)),
                      'tie_std': bool(entry.get('tie_std', False))}
    keys = list(stats)
    all_equal = all(all(np.array_equal(stats[k][a], stats[keys[0]][a]) for a in ARRAYS)
                    for k in keys)
    meta = {'format': 'unimate-stats/1', 'source': os.path.relpath(src, ROOT).replace('\\', '/'),
            'keys': keys, 'pools': pools, 'all_equal': all_equal,
            'fallback': STATS_FALLBACK, 'feature_len': int(len(stats[keys[0]]['mean_root']))}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    write_umtf(dst, tensors, meta)
    print(f'{dst}: keys={keys} all_equal={all_equal}')


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'models', 'unimate', 'unimate_uniml3d_f60_v3', 'dataset_stats.npy')
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, 'models', 'converted', 'unimate_uniml3d_f60_v3_stats.umtf')
    convert(src, dst)
