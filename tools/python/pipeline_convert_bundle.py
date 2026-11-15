"""Convert a ``data_process.rig_preprocess`` bundle to UMTF (no pickle in C++).

    python tools/python/pipeline_convert_bundle.py <bundle_dir> [out.umtf]
    python tools/python/pipeline_convert_bundle.py --examples   # eagle, shark, unitree_go2

A bundle is ``<dir>/{cond.npy, summary.json, annotation.json, <name>.glb}``
(unimate/inference/assets.py). Output (default
models/converted/bundles/<dir name>.umtf):

tensors (J joints, cond order == the model's joint order)
  parents                i32 [J]       -1 for the root
  tpos_first_frame       f32 [J,3]     canonical T-pose positions (Y up, not yet grounded)
  tpos_offsets           f32 [J,3]     offsets_from_positions(tpos): [0] = root position
  offsets                f32 [J,3]     cond 'offsets' (export-scale bone offsets; unused by sampling)
  tpos_local_rotations   f32 [J,4]     w,x,y,z
  tpos_global_rotations  f32 [J,4]     w,x,y,z (canonical rest bone frames)
  joint_relations        i32 [J,J]     edge types (self/parent/child/sibling/none/end_effector)
  joint_graph_dists      i32 [J,J]     hop distance clamped to 5
  joint_depths           i32 [J]
  edge_indexs            i64 [2,E]     stored cond copy (sampling recomputes it from parents)
  spectral_feats         f32 [J,K]     stored sign-pinned Laplacian eigenvectors (K = 8)
  <name>@f64             u8            exact float64 bytes of the float arrays above
meta
  name, joint_names (raw rig / GLB bone names), clean_joint_names,
  model_joint_names (the strings the text encoder sees: upstream
  dataset.model_joint_names rule), stats_dataset (summary.json rule of
  inference/assets.cond_file_dataset_type), scale_factor, ground_height_mode,
  face_joint_idxs, kinematic_chains, glb (file name in the bundle),
  annotation (raw/label/parent per joint, when annotation.json exists).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from umtf import write_umtf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXAMPLES = os.path.join(ROOT, 'reference', 'UniMate', 'assets', 'examples')
DATASET_TYPES = ('truebones', 'mixamo', 'objaverse', 'general')
DEFAULT_STATS_DATASET = 'objaverse'


def model_joint_names(cond):
    """upstream unimate/dataset/mixture/dataset.py::model_joint_names."""
    names = cond.get('clean_joint_names')
    n = len(cond['joint_names'])
    if names is not None and len(names) == n and all(str(x).strip() for x in names):
        return [str(x) for x in names]
    return [str(x) for x in cond['joint_names']]


def stats_dataset(summary):
    """upstream unimate/inference/assets.py::cond_file_dataset_type (summary branch)."""
    if summary is None:
        return DEFAULT_STATS_DATASET
    for key in ('stats_dataset', 'reference', 'profile'):
        if summary.get(key) in DATASET_TYPES:
            return summary[key]
    return DEFAULT_STATS_DATASET


def _f64(arr):
    return np.frombuffer(np.asarray(arr, dtype='<f8').tobytes(), dtype=np.uint8)


def find_glb(bundle_dir, name):
    files = os.listdir(bundle_dir)
    for f in files:
        if f.lower() == f'{name}.glb'.lower():
            return f
    for f in files:
        if f.lower().endswith('.glb'):
            return f
    return None


def convert(bundle_dir, dst):
    conds = np.load(os.path.join(bundle_dir, 'cond.npy'), allow_pickle=True).item()
    if len(conds) != 1:
        raise ValueError(f'{bundle_dir}/cond.npy holds {len(conds)} entries; expected 1')
    (name, cond), = conds.items()
    summary = None
    if os.path.isfile(os.path.join(bundle_dir, 'summary.json')):
        with open(os.path.join(bundle_dir, 'summary.json'), encoding='utf-8') as f:
            summary = json.load(f)
    annotation = None
    if os.path.isfile(os.path.join(bundle_dir, 'annotation.json')):
        with open(os.path.join(bundle_dir, 'annotation.json'), encoding='utf-8') as f:
            annotation = json.load(f)

    J = len(cond['parents'])
    t = {}
    t['parents'] = np.asarray(cond['parents'], dtype=np.int32)
    for k in ('tpos_first_frame', 'tpos_offsets', 'offsets',
              'tpos_local_rotations', 'tpos_global_rotations'):
        arr = np.asarray(cond[k], dtype=np.float64)
        t[k] = arr.astype(np.float32)
        t[k + '@f64'] = _f64(arr)
    t['joint_relations'] = np.asarray(cond['joint_relations'], dtype=np.int32)
    t['joint_graph_dists'] = np.asarray(cond['joint_graph_dists'], dtype=np.int32)
    t['joint_depths'] = np.asarray(cond['joint_depths'], dtype=np.int32)
    t['edge_indexs'] = np.asarray(cond['edge_indexs'], dtype=np.int64)
    t['spectral_feats'] = np.asarray(cond['spectral_feats'], dtype=np.float32)
    assert t['tpos_first_frame'].shape == (J, 3)

    sf = cond.get('scale_factor')
    meta = {
        'format': 'unimate-bundle/1',
        'name': str(name),
        'source': os.path.relpath(os.path.abspath(bundle_dir), ROOT).replace('\\', '/'),
        'num_joints': J,
        'joint_names': [str(n) for n in cond['joint_names']],
        'clean_joint_names': [str(n) for n in (cond.get('clean_joint_names') or [])],
        'model_joint_names': model_joint_names(cond),
        'stats_dataset': stats_dataset(summary),
        'scale_factor': None if sf is None else float(sf),
        'ground_height_mode': cond.get('ground_height_mode'),
        'face_joint_idxs': {k: (bool(v) if isinstance(v, (bool, np.bool_)) else int(v))
                            for k, v in (cond.get('face_joint_idxs') or {}).items()},
        'kinematic_chains': [[int(i) for i in c] for c in (cond.get('kinematic_chains') or [])],
        'glb': find_glb(bundle_dir, name),
    }
    if annotation is not None:
        meta['annotation'] = {
            'format': annotation.get('format'),
            'joints': [{'raw': j.get('raw'), 'label': j.get('label'), 'parent': j.get('parent')}
                       for j in annotation.get('joints', [])],
        }
    if summary is not None:
        meta['summary'] = {k: summary.get(k) for k in ('name', 'profile', 'stats_dataset',
                                                        'n_joints', 'canonical_glb')}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    write_umtf(dst, t, meta)
    print(f'{dst}: {name} J={J} stats_dataset={meta["stats_dataset"]} glb={meta["glb"]}')


def main(argv):
    if len(argv) >= 1 and argv[0] == '--examples':
        for d in ('eagle', 'shark', 'unitree_go2'):
            convert(os.path.join(EXAMPLES, d),
                    os.path.join(ROOT, 'models', 'converted', 'bundles', f'{d}.umtf'))
        return
    if not argv:
        print(__doc__)
        sys.exit(1)
    src = argv[0]
    dst = argv[1] if len(argv) > 1 else os.path.join(
        ROOT, 'models', 'converted', 'bundles', os.path.basename(os.path.normpath(src)) + '.umtf')
    convert(src, dst)


if __name__ == '__main__':
    main(sys.argv[1:])
