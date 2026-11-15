"""Shared helpers for denoiser_export.py / denoiser_golden.py.

Imports the read-only upstream PyTorch project (reference/UniMate) on a CPU-only
env that lacks some of its optional dependencies, builds the released denoiser
exactly as ``unimate/inference/sample.py`` does (``create_model`` +
``_load_checkpoint`` -> EMA weights), and builds a REAL conditioning dict for an
asset through the upstream ``sample._prepare_cases`` + ``create_sample_condition``
path.

Shims (installed by :func:`install_shims`, never by editing reference/):
  * ``accelerate`` (only ``set_seed`` is imported by sample.py),
  * ``Quaternions`` / ``Animation`` (Holden's Motion library, not installed):
    a small numpy re-implementation. It only feeds the rest-pose *motion*
    features (cond's ``motion`` / ``offsets``), which the denoiser never reads;
    every model-read cond key (tpos, graph tables, spectral feats, names, ...)
    comes from the unmodified upstream code. ``_validate_motion`` is stubbed
    for the same reason (its FK cross-check needs Animation).
  * ``transformers`` T5 ``from_pretrained('google/flan-t5-base')`` is redirected
    to the local copy in models/flan-t5-base.
  * PIL / matplotlib plotting (``unimate.utils.visualization``) is stubbed.
"""
import os
import sys
import types
import importlib.machinery

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPSTREAM = os.path.join(ROOT, "reference", "UniMate")
MODEL_NAME = "unimate_uniml3d_f60_v3"
MODEL_DIR = os.path.join(ROOT, "models", "unimate", MODEL_NAME)
CKPT = os.path.join(MODEL_DIR, "checkpoints", "checkpoint_step_150000.pt")
T5_DIR = os.path.join(ROOT, "models", "flan-t5-base")
CONVERTED = os.path.join(ROOT, "models", "converted", MODEL_NAME + ".umtf")
GOLDEN_DIR = os.path.join(ROOT, "tests", "golden", "denoiser")

_installed = False


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


# ---------------------------------------------------------------------------
# Minimal Holden-style Quaternions ([w, x, y, z], Hamilton product, -q = conj)
# ---------------------------------------------------------------------------
class Quaternions:
    def __init__(self, qs):
        if isinstance(qs, Quaternions):
            qs = qs.qs
        self.qs = np.array(qs, dtype=np.float64)

    shape = property(lambda self: self.qs.shape[:-1])
    ndim = property(lambda self: self.qs.ndim - 1)

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, k):
        return Quaternions(self.qs[k])

    def __setitem__(self, k, v):
        self.qs[k] = v.qs if isinstance(v, Quaternions) else v

    def copy(self):
        return Quaternions(self.qs.copy())

    def repeat(self, n, axis=None):  # np.repeat(q, ...) dispatches here
        return Quaternions(np.repeat(self.qs, n, axis=axis))

    def __neg__(self):
        return Quaternions(self.qs * np.array([1.0, -1.0, -1.0, -1.0]))

    @staticmethod
    def _mul(a, b):
        aw, ax, ay, az = np.moveaxis(a, -1, 0)
        bw, bx, by, bz = np.moveaxis(b, -1, 0)
        return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                         aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw], axis=-1)

    def __mul__(self, other):
        if isinstance(other, Quaternions):
            return Quaternions(self._mul(self.qs, other.qs))
        v = np.asarray(other, dtype=np.float64)
        if v.shape[-1] == 3:  # rotate vectors
            vq = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
            conj = self.qs * np.array([1.0, -1.0, -1.0, -1.0])
            return self._mul(self._mul(self.qs, vq), conj)[..., 1:]
        raise TypeError("unsupported Quaternions product")

    def rotation_matrix(self, cont6d=False):
        q = self.qs / np.linalg.norm(self.qs, axis=-1, keepdims=True)
        w, x, y, z = np.moveaxis(q, -1, 0)
        m = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                     axis=-1).reshape(q.shape[:-1] + (3, 3))
        if cont6d:  # first two columns, matching unimate rotation_conversions
            return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)
        return m

    @classmethod
    def id(cls, n):
        qs = np.zeros((n, 4) if np.isscalar(n) else tuple(n) + (4,))
        qs[..., 0] = 1.0
        return cls(qs)


def _offsets_from_positions(positions, parents):
    parents = np.asarray(parents)
    offsets = positions - positions[..., np.maximum(parents, 0), :]
    offsets[..., 0, :] = positions[..., 0, :]
    return offsets


def install_shims():
    """Make ``import unimate...`` work on this CPU-only env (idempotent)."""
    global _installed
    if _installed:
        return
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    # transformers probes optional packages at import time; import it before
    # the accelerate stub exists so it does not take the stub for the real one.
    import transformers  # noqa: F401
    from transformers import T5EncoderModel, T5Tokenizer

    _stub("accelerate")
    _stub("accelerate.utils", set_seed=lambda seed: None)
    _stub("Quaternions", Quaternions=Quaternions)
    _stub("Animation", offsets_from_positions=_offsets_from_positions)
    try:
        import PIL  # noqa: F401
    except ImportError:
        _stub("unimate.utils.visualization",
              visualize_and_save_motions=lambda *a, **k: None)

    # Local flan-t5-base instead of the HF hub id.
    def _local(fn):
        def wrapped(name, *a, **k):
            if name == "google/flan-t5-base":
                name = T5_DIR
            return fn(name, *a, **k)
        return wrapped

    import unimate.models.text_encoder.t5 as t5mod
    t5mod.T5Tokenizer = types.SimpleNamespace(
        from_pretrained=_local(T5Tokenizer.from_pretrained))
    t5mod.T5EncoderModel = types.SimpleNamespace(
        from_pretrained=_local(T5EncoderModel.from_pretrained))

    # The rest-pose clip's FK self-check needs the Animation library; the
    # motion it validates is never read by the denoiser.
    import unimate.dataset.mixture.dataset as dsmod
    dsmod._validate_motion = lambda *a, **k: (True, "")
    _installed = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_config():
    install_shims()
    from unimate.configs.schema import MainConfig
    return MainConfig.from_json(os.path.join(MODEL_DIR, "config.json"))


def load_model(config=None):
    """Upstream model with the weights inference uses (sample.py: create_model
    + _load_checkpoint, i.e. EMA shadow params copied over the raw weights)."""
    install_shims()
    import torch
    from unimate.models.factory import create_model
    from unimate.inference import sample as sample_mod
    config = config or load_config()
    model = create_model(dataset_config=config.dataset, model_config=config.model)
    sample_mod._load_checkpoint(model, CKPT, config)
    model.eval()
    torch.set_grad_enabled(False)
    return model, config


# ---------------------------------------------------------------------------
# Conditioning (sample.py --asset <dir> --prompt <text>)
# ---------------------------------------------------------------------------
def build_cond(config, cases, seed=0):
    """``cases``: list of (asset_dir relative to reference/UniMate, prompt).

    Mirrors sample.main: _prepare_cases (asset resolution, skeleton dataset,
    T5 prompt + joint-name encoding) then create_sample_condition over the
    whole list as one chunk. Returns the upstream cond dict (torch tensors).
    """
    install_shims()
    import random
    import torch
    from unimate.inference import sample as sample_mod
    from unimate.dataset.conditioning import create_sample_condition

    cwd = os.getcwd()
    os.chdir(UPSTREAM)  # sample.py is run from the repo root
    try:
        assets = tuple(dict.fromkeys(a for a, _ in cases))
        prompts = tuple(dict.fromkeys(p for _, p in cases))
        args = sample_mod.InferenceArgs(exp_dir=MODEL_DIR, asset=assets, prompt=prompts)
        cfg_scale = config.sampling.cfg_scale
        stats_path = sample_mod._resolve_stats_path(MODEL_DIR)
        dataset, enc_cases, _ = sample_mod._prepare_cases(
            args, config, cfg_scale, stats_path, torch.device("cpu"))
        # keep (asset, prompt) pairs in the requested order
        by_key = {}
        for case_id, name, caption, enc, clip in enc_cases:
            by_key.setdefault((name, caption), (name, caption, enc))
        names = {a: sample_mod.AssetResolver(config.dataset).resolve(a).name for a in assets}
        tcc = [by_key[(names[a], p)] for a, p in cases]
        random.seed(seed)
        np.random.seed(seed)
        _, cond = create_sample_condition(config=config, data=dataset,
                                          test_case_captions=tcc)
    finally:
        os.chdir(cwd)
    return cond
