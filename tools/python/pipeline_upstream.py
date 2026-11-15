"""Run the upstream UniMate (PyTorch) inference code on CPU from this repo.

Importing this module:
  * puts reference/UniMate on sys.path (read-only reference, never edited);
  * installs minimal stand-ins for packages that are not installed here and
    are irrelevant to (or only re-implemented for) the text-to-motion path:
      - ``Quaternions`` / ``Animation``: the Holden-style Motion library
        (git dependency of upstream). Re-implemented below with the same
        conventions (w,x,y,z Hamilton product, q*v = q v q^-1, -q = conjugate,
        transforms() = standard rotation matrix, rotation_matrix(cont6d=True)
        = first two matrix COLUMNS, positions_global ignores ``orients``,
        offsets_from_positions keeps the root position as offsets[0]).
        These conventions are pinned by upstream code: the cont6d encoding
        must round-trip with unimate.utils.rotation_conversions (columns),
        and ``tpos_offsets[0] == tpos_first_frame[0]`` in every shipped cond.
      - ``accelerate`` (only set_seed), ``clip``, ``PIL``,
        ``data_process.utils.plotting`` (rendering only).
  * makes ``T5Tokenizer/T5EncoderModel.from_pretrained('google/flan-t5-base')``
    load the local copy in models/flan-t5-base (offline).
"""
import os
import sys
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPSTREAM = os.path.join(ROOT, 'reference', 'UniMate')
MODEL_DIR = os.path.join(ROOT, 'models', 'unimate', 'unimate_uniml3d_f60_v3')
T5_DIR = os.path.join(ROOT, 'models', 'flan-t5-base')
EXAMPLES = os.path.join(UPSTREAM, 'assets', 'examples')

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)


# ---------------------------------------------------------------------------
# Quaternions (Holden Motion library subset)
# ---------------------------------------------------------------------------
class Quaternions:
    def __init__(self, qs):
        if isinstance(qs, Quaternions):
            self.qs = qs.qs
            return
        qs = np.asarray(qs)
        if qs.ndim == 1:
            qs = np.array([qs])
        self.qs = qs

    def __str__(self):
        return 'Quaternions(' + str(self.qs) + ')'

    __repr__ = __str__

    @property
    def shape(self):
        return self.qs.shape[:-1]

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, k):
        return Quaternions(self.qs[k])

    def __setitem__(self, k, v):
        self.qs[k] = v.qs if isinstance(v, Quaternions) else v

    def copy(self):
        return Quaternions(np.copy(self.qs))

    def repeat(self, n, **kwargs):
        return Quaternions(self.qs.repeat(n, **kwargs))

    @property
    def real(self):
        return self.qs[..., 0]

    @property
    def imaginaries(self):
        return self.qs[..., 1:4]

    def __neg__(self):
        return Quaternions(self.qs * np.array([1, -1, -1, -1]))

    def __mul__(self, other):
        if isinstance(other, Quaternions):
            q, r = self.qs, other.qs
            q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            r0, r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
            return Quaternions(np.stack([
                r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3,
                r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2,
                r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1,
                r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0], axis=-1))
        other = np.asarray(other)
        if other.shape[-1] == 3:
            vs = Quaternions(np.concatenate([np.zeros(other.shape[:-1] + (1,)), other], axis=-1))
            return (self * (vs * -self)).imaginaries
        raise TypeError('unsupported Quaternions product')

    def normalized(self):
        return Quaternions(self.qs / np.linalg.norm(self.qs, axis=-1)[..., None])

    def transforms(self):
        qw, qx, qy, qz = self.qs[..., 0], self.qs[..., 1], self.qs[..., 2], self.qs[..., 3]
        x2, y2, z2 = qx + qx, qy + qy, qz + qz
        xx, yy, wx = qx * x2, qy * y2, qw * x2
        xy, yz, wy = qx * y2, qy * z2, qw * y2
        xz, zz, wz = qx * z2, qz * z2, qw * z2
        m = np.empty(self.shape + (3, 3))
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)
        return m

    def rotation_matrix(self, cont6d=False):
        m = self.transforms()
        if cont6d:
            return np.concatenate([m[..., 0], m[..., 1]], axis=-1)
        return m

    @classmethod
    def id(cls, n):
        if isinstance(n, tuple):
            qs = np.zeros(n + (4,))
        else:
            qs = np.zeros((n, 4))
        qs[..., 0] = 1.0
        return Quaternions(qs)

    @classmethod
    def id_like(cls, a):
        return cls.id(a.shape)


# ---------------------------------------------------------------------------
# Animation (Holden Motion library subset)
# ---------------------------------------------------------------------------
class Animation:
    def __init__(self, rotations, positions, orients, offsets, parents):
        self.rotations = rotations
        self.positions = positions
        self.orients = orients
        self.offsets = offsets
        self.parents = np.asarray(parents)

    @property
    def shape(self):
        return (self.rotations.shape[0], self.rotations.shape[1])

    def copy(self):
        return Animation(self.rotations.copy(), self.positions.copy(), self.orients.copy(),
                         self.offsets.copy(), self.parents.copy())


def transforms_multiply(t0s, t1s):
    return np.matmul(t0s, t1s)


def transforms_blank(anim):
    ts = np.zeros(anim.shape + (4, 4))
    ts[..., 0, 0] = ts[..., 1, 1] = ts[..., 2, 2] = ts[..., 3, 3] = 1.0
    return ts


def transforms_local(anim):
    t = anim.rotations.transforms()
    t = np.concatenate([t, np.zeros(t.shape[:2] + (3, 1))], axis=-1)
    t = np.concatenate([t, np.zeros(t.shape[:2] + (1, 4))], axis=-2)
    t[:, :, 0:3, 3] = anim.positions
    t[:, :, 3:4, 3] = 1.0
    return t


def transforms_global(anim):
    loc = transforms_local(anim)
    glob = transforms_blank(anim)
    glob[:, 0] = loc[:, 0]
    for i in range(1, anim.shape[1]):
        glob[:, i] = transforms_multiply(glob[:, anim.parents[i]], loc[:, i])
    return glob


def positions_global(anim):
    p = transforms_global(anim)[:, :, :, 3]
    return p[:, :, :3] / p[:, :, 3, np.newaxis]


def offsets_from_positions(positions, parents):
    positions = np.asarray(positions)
    offsets = np.array(positions, dtype=np.float64, copy=True)
    for j in range(1, len(parents)):
        offsets[j] = positions[j] - positions[parents[j]]
    return offsets


def _install_shims():
    qmod = types.ModuleType('Quaternions')
    qmod.Quaternions = Quaternions
    sys.modules.setdefault('Quaternions', qmod)

    amod = types.ModuleType('Animation')
    for name in ('Animation', 'Quaternions', 'transforms_local', 'transforms_global',
                 'transforms_multiply', 'transforms_blank', 'positions_global',
                 'offsets_from_positions'):
        setattr(amod, name, globals()[name])
    sys.modules.setdefault('Animation', amod)

    def _set_seed(seed, *a, **k):
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    acc = types.ModuleType('accelerate')
    accu = types.ModuleType('accelerate.utils')
    accu.set_seed = _set_seed
    acc.utils = accu
    sys.modules.setdefault('accelerate', acc)
    sys.modules.setdefault('accelerate.utils', accu)

    sys.modules.setdefault('clip', types.ModuleType('clip'))
    pil = types.ModuleType('PIL')
    pil.Image = types.ModuleType('PIL.Image')
    pil.Image.Image = type('Image', (), {})
    sys.modules.setdefault('PIL', pil)
    sys.modules.setdefault('PIL.Image', pil.Image)

    plot = types.ModuleType('data_process.utils.plotting')
    for name in ('save_skeleton_motion', 'save_skeleton_motion_spectral',
                 'render_skeleton_tpose', 'render_skeleton_tpos_spectral'):
        setattr(plot, name, lambda *a, **k: None)
    sys.modules.setdefault('data_process.utils.plotting', plot)


def _patch_t5():
    import transformers
    for cls_name in ('T5Tokenizer', 'T5EncoderModel'):
        cls = getattr(transformers, cls_name)
        orig = cls.from_pretrained.__func__ if hasattr(cls.from_pretrained, '__func__') else None

        def make(orig_fn, c):
            def from_pretrained(klass, name, *a, **k):
                if name in ('google/flan-t5-base',):
                    name = T5_DIR
                return orig_fn(klass, name, *a, **k)
            return classmethod(from_pretrained)

        if orig is not None:
            setattr(cls, 'from_pretrained', make(orig, cls))


_patch_t5()
_install_shims()


def load_bundle_cond(bundle_dir):
    """(name, cond_entry, summary, annotation) of a rig_preprocess bundle."""
    import json
    conds = np.load(os.path.join(bundle_dir, 'cond.npy'), allow_pickle=True).item()
    assert len(conds) == 1, f'{bundle_dir}: expected one cond entry'
    (name, cond), = conds.items()
    with open(os.path.join(bundle_dir, 'summary.json')) as f:
        summary = json.load(f)
    ann_path = os.path.join(bundle_dir, 'annotation.json')
    annotation = None
    if os.path.isfile(ann_path):
        with open(ann_path) as f:
            annotation = json.load(f)
    return name, cond, summary, annotation


def find_glb(bundle_dir, name):
    """The bundle's canonical GLB: ``<name>.glb`` (case-insensitive)."""
    for f in os.listdir(bundle_dir):
        if f.lower() == f'{name}.glb'.lower():
            return os.path.join(bundle_dir, f)
    for f in os.listdir(bundle_dir):
        if f.lower().endswith('.glb'):
            return os.path.join(bundle_dir, f)
    return None
