"""Goldens for src/pipeline from the real upstream UniMate code, run on CPU.

    python tools/python/pipeline_golden.py [--assets eagle shark] [--skip-model]

Writes tests/golden/pipeline/:

  ode_analytic.umtf      torchdiffeq on analytic vector fields (same formulas as
                         tests/pipeline: field_rf / field_osc / field_tanh):
                         dopri5 (sample_ode defaults: t = linspace(0,1,50),
                         atol 1e-6, rtol 1e-3), euler and heun2 (num_steps 20).
  <asset>_cond.umtf      the model cond dict exactly as create_sample_condition
                         builds it for ``--asset <bundle> --prompt P`` (B = 1;
                         list-valued keys stacked: parents [1,J], edge_indexs
                         [1,2,E]); the strings fed to T5 and their embeddings.
  <asset>_sample.umtf    initial noise, Euler-20 and default dopri5 samples with
                         every velocity call (t, v) for replay, NFE, accepted /
                         rejected steps, initial step.
  <asset>_motion.umtf    de-normalised features, recovered root / BVH rotations /
                         FK and RIC positions, the Blender keyframes
                         (compute_bone_keyframes) and the resulting glTF
                         node-local tracks + world joint positions.

Ground truth vs re-derived: everything up to ``recover_unimate_anim_from_rot``
is the upstream code itself (Motion library = pipeline_upstream shim).
compute_bone_keyframes is upstream math re-run in numpy (mathutils
Matrix.to_quaternion == rotation matrix -> quaternion). The Blender part --
glTF import of the canonical GLB, motion_axis_alignment (assumed None: the
imported bones keep the baked tpos_global_rotations axes), pose evaluation and
glTF export -- is re-derived here in numpy, NOT ground truth.
"""
import argparse
import json
import math
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import pipeline_upstream as U  # noqa: E402  (installs shims, sys.path)
from umtf import write_umtf  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(U.ROOT, 'tests', 'golden', 'pipeline')
PROMPTS = {
    'eagle': 'An object flaps its wings and takes off.',
    'shark': 'An object swims forward.',
    'unitree_go2': 'An object walks forward.',
}

torch.set_num_threads(max(1, os.cpu_count() or 1))


# ---------------------------------------------------------------------------
# step recorder for torchdiffeq's adaptive solver
# ---------------------------------------------------------------------------
class StepRecorder:
    def __init__(self):
        from torchdiffeq._impl import rk_common
        self.rk = rk_common
        self.orig_step = rk_common.RKAdaptiveStepsizeODESolver._adaptive_step
        self.orig_init = rk_common._select_initial_step
        self.reset()

    def reset(self):
        self.steps = []          # (t0, dt, accepted, error_ratio)
        self.first_step = None

    def __enter__(self):
        rec = self
        orig_step, orig_init = self.orig_step, self.orig_init

        def init(*a, **k):
            h = orig_init(*a, **k)
            rec.first_step = float(h)
            return h

        def step(solver, rk_state):
            t0 = float(rk_state.t1)
            dt_req = rk_state.dt
            new = orig_step(solver, rk_state)
            accepted = float(new.t1) != t0
            dt = float(new.t1) - t0 if accepted else float(
                dt_req.clamp(solver.min_step, solver.max_step))
            rec.steps.append((t0, dt, accepted))
            return new

        self.rk._select_initial_step = init
        self.rk.RKAdaptiveStepsizeODESolver._adaptive_step = step
        return self

    def __exit__(self, *exc):
        self.rk._select_initial_step = self.orig_init
        self.rk.RKAdaptiveStepsizeODESolver._adaptive_step = self.orig_step


def ode_solve(fn, x0, method, num_steps):
    """integrators.ode.sample as upstream calls it (t0=0, t1=1 for velocity /
    linear: Transport.check_interval)."""
    from unimate.models.flow.integrators import ode
    calls = []

    def drift(x, t, model, **kw):
        v = fn(x, t)
        calls.append((float(t[0]), v.detach().clone()))
        return v

    o = ode(drift=drift, t0=0, t1=1, sampler_type=method, num_steps=num_steps,
            atol=1e-6, rtol=1e-3)
    with StepRecorder() as rec:
        out = o.sample(x0, None)[-1]
    return out, calls, rec


# ---------------------------------------------------------------------------
# analytic vector fields (mirrored in tests/pipeline/test_sampler.cpp)
# ---------------------------------------------------------------------------
def analytic_inputs():
    shape = (1, 4, 3, 5)
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    x0 = np.sin(0.7 * i + 0.3) * 1.3              # deterministic "noise"
    mu = np.cos(0.37 * i) * 0.8
    a = 0.5 + 0.5 * np.sin(1.1 * i)
    b = 0.3 * np.cos(0.9 * i + 1.0)
    f = lambda v: torch.tensor(v.reshape(shape), dtype=torch.float32)  # noqa: E731
    return shape, f(x0), f(mu), f(a), f(b)


def fields(mu, a, b):
    def field_rf(x, t):
        tt = t.view(-1, 1, 1, 1).to(x.dtype)
        return (mu - x) / (1.1 - tt)

    def field_osc(x, t):
        tt = t.view(-1, 1, 1, 1).to(x.dtype)
        return torch.cos(4.0 * tt) * a * x + b * torch.sin(x)

    def field_tanh(x, t):
        tt = t.view(-1, 1, 1, 1).to(x.dtype)
        return torch.tanh(x) * (1.0 - tt) * 3.0 - 0.5 * x + mu * tt

    def field_stiff(x, t):
        tt = t.view(-1, 1, 1, 1).to(x.dtype)
        return 8.0 * torch.cos(11.0 * tt) * a - 12.0 * (x - mu * torch.sin(7.0 * tt)) \
            + 4.0 * torch.sin(3.0 * x)

    return {'rf': field_rf, 'osc': field_osc, 'tanh': field_tanh, 'stiff': field_stiff}


def make_ode_goldens():
    shape, x0, mu, a, b = analytic_inputs()
    tensors = {'x0': x0, 'mu': mu, 'a': a, 'b': b}
    meta = {'shape': list(shape), 'fields': {}}
    for name, fn in fields(mu, a, b).items():
        m = {}
        out, calls, rec = ode_solve(fn, x0, 'dopri5', 50)
        tensors[f'{name}/dopri5'] = out
        steps = np.array([[s[0], s[1], float(s[2])] for s in rec.steps], dtype=np.float64)
        tensors[f'{name}/dopri5_steps@f64'] = np.frombuffer(steps.astype('<f8').tobytes(), np.uint8)
        tensors[f'{name}/dopri5_steps'] = steps.astype(np.float32)
        m['dopri5_nfe'] = len(calls)
        m['dopri5_first_step'] = rec.first_step
        m['dopri5_accepted'] = int(steps[:, 2].sum())
        m['dopri5_rejected'] = int(len(steps) - steps[:, 2].sum())
        for method, key in (('euler', 'euler20'), ('heun2', 'heun20')):
            out, calls, _ = ode_solve(fn, x0, method, 20)
            tensors[f'{name}/{key}'] = out
            m[f'{key}_nfe'] = len(calls)
        meta['fields'][name] = m
        print(f'[ode] {name}: dopri5 nfe={m["dopri5_nfe"]} acc={m["dopri5_accepted"]} '
              f'rej={m["dopri5_rejected"]} h0={m["dopri5_first_step"]:.6g}')
    write_umtf(os.path.join(OUT, 'ode_analytic.umtf'), tensors, meta)


# ---------------------------------------------------------------------------
# upstream model path
# ---------------------------------------------------------------------------
def build_upstream(asset_dir, prompt):
    """Mirror unimate.inference.sample.main up to the cond dict (CPU)."""
    from unimate.configs.schema import MainConfig
    from unimate.inference import sample as S

    config = MainConfig.from_json(os.path.join(U.MODEL_DIR, 'config.json'))
    config.sampling.device = 'cpu'
    args = S.InferenceArgs(exp_dir=U.MODEL_DIR, asset=(asset_dir,), prompt=(prompt,), seed=10)
    cfg_scale = config.sampling.cfg_scale
    S._validate_cfg_inputs(cfg_scale, args)
    sys.modules['accelerate.utils'].set_seed(10)

    # record every string the T5 encoder sees and what it returns
    from unimate.models.text_encoder import t5 as t5mod
    seen = []
    orig_tok, orig_fwd = t5mod.T5Conditioner.tokenize, t5mod.T5Conditioner.forward

    def tok(self, x):
        inputs = orig_tok(self, x)
        inputs['_texts'] = [x] if isinstance(x, str) else list(x)
        return inputs

    def fwd(self, inputs):
        texts = inputs.pop('_texts', None)
        h = orig_fwd(self, inputs)
        seen.append((texts, inputs['attention_mask'].detach().clone(), h.detach().clone()))
        return h

    t5mod.T5Conditioner.tokenize, t5mod.T5Conditioner.forward = tok, fwd
    try:
        stats_path = S._resolve_stats_path(U.MODEL_DIR)
        dataset, cases, assets = S._prepare_cases(args, config, cfg_scale, stats_path,
                                                  torch.device('cpu'))
    finally:
        t5mod.T5Conditioner.tokenize, t5mod.T5Conditioner.forward = orig_tok, orig_fwd

    from unimate.dataset.conditioning import create_sample_condition
    case = cases[0]
    _, cond = create_sample_condition(config=config, data=dataset,
                                      test_case_captions=[(case[1], case[2], case[3])])
    md = dataset.motion_dataset
    return config, cfg_scale, dataset, md, case, cond, seen, assets


def cond_to_arrays(cond):
    out = {}
    for k, v in cond.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy()
        elif k == 'parents':
            out[k] = np.stack([np.asarray(p, dtype=np.int64) for p in v])
        elif k == 'edge_indexs':
            out[k] = np.stack([np.asarray(e, dtype=np.int64) for e in v])
    return out


def load_model(config):
    from unimate.inference import sample as S
    from unimate.models.factory import create_model
    model = create_model(dataset_config=config.dataset, model_config=config.model)
    diffusion, gen = S._build_diffusion(config)
    _, model_path = S._resolve_exp_paths(U.MODEL_DIR, None)
    S._load_checkpoint(model, model_path, config)
    model.eval()
    return model, diffusion, gen


def run_asset(name, euler_only=False):
    asset_dir = os.path.join(U.EXAMPLES, name)
    prompt = PROMPTS[name]
    print(f'== {name}: {prompt!r}')
    config, cfg_scale, dataset, md, case, cond, seen, assets = build_upstream(asset_dir, prompt)
    obj_type = case[1]

    # --- strings fed to T5 -------------------------------------------------
    joint_calls = [s for s in seen if s[0] is not None and s[0] != [prompt]]
    cap_calls = [s for s in seen if s[0] == [prompt]]
    assert len(joint_calls) == 1 and len(cap_calls) == 1, [s[0] for s in seen]
    jn_texts = joint_calls[0][0]
    jn_emb = np.stack([md._joint_name_emb[t] for t in jn_texts]).astype(np.float32)
    cap_emb = np.asarray(case[3]['caption_emb'], dtype=np.float32)
    cap_tok = np.asarray(case[3]['caption_tokens'], dtype=np.float32)
    model_names = [str(n) for n in md.cond_dict[obj_type].get('clean_joint_names')]

    arrays = cond_to_arrays(cond)
    tensors = {f'cond/{k}': v for k, v in arrays.items()}
    tensors['joint_name_text_emb'] = jn_emb
    tensors['caption_emb'] = cap_emb
    tensors['caption_tokens'] = cap_tok
    meta = {
        'asset': name, 'object_type': obj_type, 'prompt': prompt,
        'joint_name_texts': jn_texts, 'model_joint_names': model_names,
        'caption_texts': cap_calls[0][0],
        'cond_keys': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()},
        'cond_non_tensor': {k: (v if isinstance(v, list) and all(isinstance(e, str) for e in v)
                                else None) for k, v in cond.items()
                            if not torch.is_tensor(v) and k not in ('parents', 'edge_indexs')},
        'stats_key': md.resolve_stats_key(assets[obj_type].stats_dataset),
        'stats_dataset': assets[obj_type].stats_dataset,
        'max_joints': config.dataset.max_joints, 'max_frames': config.dataset.max_motion_length,
        'feature_len': config.dataset.feature_len, 'max_depth': config.dataset.max_depth,
        'max_freqs': config.model.max_freqs, 'cfg_scale': cfg_scale,
    }
    write_umtf(os.path.join(OUT, f'{name}_cond.umtf'), tensors, meta)
    print(f'[cond] {len(arrays)} keys, {len(jn_texts)} unique joint names')

    # --- sampling ----------------------------------------------------------
    from unimate.inference.generate import ClassifierFreeSampleModel
    model, diffusion, gen = load_model(config)
    cfg_model = ClassifierFreeSampleModel(model, cfg_scale=cfg_scale)
    shape = (1, config.dataset.max_joints, config.dataset.feature_len,
             config.dataset.max_motion_length)
    torch.manual_seed(1234)
    noise = torch.randn(shape)

    def vel(x, t):
        with torch.no_grad():
            return cfg_model(x, t, cond=cond)

    s_t = {'noise': noise}
    s_meta = {'asset': name, 'cfg_scale': cfg_scale, 'noise_seed_torch': 1234}
    runs = [('euler20', 'euler', 20)] + ([] if euler_only else [('dopri5', 'dopri5', 50)])
    for key, method, n in runs:
        t0 = time.time()
        out, calls, rec = ode_solve(vel, noise, method, n)
        dt = time.time() - t0
        s_t[key] = out
        s_t[f'{key}_call_t'] = np.array([c[0] for c in calls], dtype=np.float32)
        s_t[f'{key}_call_v'] = torch.stack([c[1][0] for c in calls])
        s_meta[f'{key}_nfe'] = len(calls)
        s_meta[f'{key}_seconds'] = dt
        if method == 'dopri5':
            steps = np.array([[s[0], s[1], float(s[2])] for s in rec.steps], dtype=np.float64)
            s_t['dopri5_steps'] = steps.astype(np.float32)
            s_t['dopri5_steps@f64'] = np.frombuffer(steps.astype('<f8').tobytes(), np.uint8)
            s_meta['dopri5_first_step'] = rec.first_step
            s_meta['dopri5_accepted'] = int(steps[:, 2].sum())
            s_meta['dopri5_rejected'] = int(len(steps) - steps[:, 2].sum())
        print(f'[sample] {key}: nfe={len(calls)} ({dt:.1f}s)')
    final_key = 'dopri5' if 'dopri5' in s_t else 'euler20'
    s_meta['final'] = final_key
    write_umtf(os.path.join(OUT, f'{name}_sample.umtf'), s_t, s_meta)

    # --- de-normalise + recover -------------------------------------------
    m_t, m_meta = recover_and_animate(name, asset_dir, md, obj_type, cond, s_t[final_key])
    m_meta['sample_key'] = final_key
    write_umtf(os.path.join(OUT, f'{name}_motion.umtf'), m_t, m_meta)


# ---------------------------------------------------------------------------
# recovery (upstream) + Blender-side re-derivation (numpy)
# ---------------------------------------------------------------------------
def quat_to_mat(q):
    return U.Quaternions(np.asarray(q, dtype=np.float64)).transforms()


def mat_to_quat_blender(m):
    """mathutils Matrix.to_quaternion for an orthonormal 3x3 (w,x,y,z, w>=0)."""
    from unimate.utils.rotation_conversions import matrix_to_quaternion_np
    return matrix_to_quaternion_np(m)


def read_glb_json(path):
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    return json.loads(b[20:20 + n])


def node_local(n):
    M = np.eye(4)
    if 'matrix' in n:
        return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
    t = np.array(n.get('translation', [0, 0, 0]), dtype=np.float64)
    r = np.array(n.get('rotation', [0, 0, 0, 1]), dtype=np.float64)
    s = np.array(n.get('scale', [1, 1, 1]), dtype=np.float64)
    R = quat_to_mat(np.array([r[3], r[0], r[1], r[2]]))
    M[:3, :3] = R * s[None, :]
    M[:3, 3] = t
    return M


def glb_rest(path):
    j = read_glb_json(path)
    nodes = j['nodes']
    parent = [-1] * len(nodes)
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i
    world = [None] * len(nodes)

    def w(i):
        if world[i] is None:
            world[i] = node_local(nodes[i]) if parent[i] < 0 else w(parent[i]) @ node_local(nodes[i])
        return world[i]

    for i in range(len(nodes)):
        w(i)
    return nodes, parent, np.stack(world)


def umeyama(X, Y):
    """data_process/utils/skeleton.py::similarity_fit (copied: that module imports bpy)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    Uu, Ss, Vt = np.linalg.svd(Yc.T @ Xc)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Uu @ Vt))
    R = Uu @ D @ Vt
    s = float((Ss * D.diagonal()).sum() / max((Xc ** 2).sum(), 1e-30))
    t = my - s * R @ mx
    return R, s, t


def recover_and_animate(name, asset_dir, md, obj_type, cond, sample):
    from unimate.utils import motion_utils as MU
    cond_entry = md.cond_dict[obj_type]
    nj = int(cond['n_joints'][0])
    parents = np.asarray(cond['parents'][0])
    # visualize_and_save_motions: f32 de-normalisation
    motion = sample[0, :nj].cpu().permute(2, 0, 1).numpy().copy()
    mean = cond['mean'][0][:nj][None].cpu().numpy()
    std = cond['std'][0][:nj][None].cpu().numpy()
    motion = motion * std + mean                              # (T, J, 12) f32
    offsets_vis = cond['offsets'][0][:nj].cpu().numpy()

    r_rot, r_pos = MU.recover_unimate_root_quat_and_pos(motion[:, 0])
    pos_ric = MU.recover_unimate_joint_pos_from_ric(motion)
    pos_fk_vis = MU.recover_unimate_joint_pos_from_rot(motion, parents, offsets_vis)

    # animate_motion.build_anim_from_npy, canonical asset (scale_factor = 1)
    tpos_offsets = np.asarray(cond_entry['tpos_offsets'], dtype=np.float64)
    anim = MU.recover_unimate_anim_from_rot(motion, parents, tpos_offsets)
    bvh = anim.rotations.qs                                   # (T, J, 4) wxyz
    pos_anim = U.positions_global(anim)                       # (T, J, 3)
    T = motion.shape[0]

    # ---- compute_bone_keyframes (upstream math) ----
    rest = U.Animation(U.Quaternions.id((1, nj)), tpos_offsets[None], U.Quaternions.id(nj),
                       tpos_offsets, parents)
    anim_local = U.transforms_local(anim)
    rest_local = U.transforms_local(rest)[0]
    rel = np.linalg.inv(rest_local)[None] @ anim_local
    tg = np.asarray(cond_entry['tpos_global_rotations'], dtype=np.float64)   # wxyz
    rel_q = U.Quaternions(mat_to_quat_blender(rel[..., :3, :3]))
    # tq^-1 @ rot @ tq (mathutils @ is the Hamilton product)
    tq = U.Quaternions(np.broadcast_to(tg[None], (T, nj, 4)).copy())
    key_rot = (-tq) * rel_q * tq
    key_loc = (-tq) * rel[..., :3, 3]

    # ---- Blender pose evaluation with the baked rest bones ----
    tpos = np.asarray(cond_entry['tpos_first_frame'], dtype=np.float64)
    B = np.zeros((nj, 4, 4))
    B[:, :3, :3] = quat_to_mat(tg)
    B[:, :3, 3] = tpos
    B[:, 3, 3] = 1
    basis = np.zeros((T, nj, 4, 4))
    basis[..., :3, :3] = quat_to_mat(key_rot.qs)
    basis[..., :3, 3] = key_loc
    basis[..., 3, 3] = 1
    M = np.zeros((T, nj, 4, 4))
    for j in range(nj):
        p = parents[j]
        if p < 0:
            M[:, j] = B[j] @ basis[:, j]
        else:
            M[:, j] = M[:, p] @ (np.linalg.inv(B[p]) @ B[j]) @ basis[:, j]
    D = M @ np.linalg.inv(B)[None]                           # armature-space world delta

    # ---- glTF export re-derivation on the bundle's GLB ----
    glb = U.find_glb(asset_dir, name if os.path.isfile(os.path.join(asset_dir, f'{name}.glb')) else obj_type)
    nodes, nparent, rest_world = glb_rest(glb)
    names = [n.get('name') for n in nodes]
    jn = [str(n) for n in cond_entry['joint_names']]
    idx = [names.index(n) for n in jn]
    Wr = rest_world[idx]
    R_s, s_s, t_s = umeyama(tpos, Wr[:, :3, 3])
    S = np.eye(4)
    S[:3, :3] = s_s * R_s
    S[:3, 3] = t_s
    Sinv = np.linalg.inv(S)
    world = np.einsum('ab,tjbc,cd,jde->tjae', S, D, Sinv, Wr)  # (T, J, 4, 4)
    # node-local TRS (joints only; non-joint ancestors keep their rest)
    loc_t = np.zeros((T, nj, 3))
    loc_r = np.zeros((T, nj, 4))
    loc_s = np.zeros((T, nj, 3))
    jset = {g: k for k, g in enumerate(idx)}
    for k, g in enumerate(idx):
        p = nparent[g]
        if p in jset:
            PW = world[:, jset[p]]
        elif p >= 0:
            PW = np.broadcast_to(rest_world[p], (T, 4, 4))
        else:
            PW = np.broadcast_to(np.eye(4), (T, 4, 4))
        L = np.linalg.inv(PW) @ world[:, k]
        sc = np.linalg.norm(L[:, :3, :3], axis=1)
        Rm = L[:, :3, :3] / sc[:, None, :]
        q = mat_to_quat_blender(Rm)                           # wxyz
        loc_t[:, k] = L[:, :3, 3]
        loc_r[:, k] = q[:, [1, 2, 3, 0]]                      # glTF xyzw
        loc_s[:, k] = sc
    world_pos = world[..., :3, 3]
    tensors = {
        'motion': motion.astype(np.float32),
        'root_rot_quat': r_rot.qs.astype(np.float32),
        'root_pos': r_pos.astype(np.float32),
        'positions_ric': pos_ric.astype(np.float32),
        'positions_fk': pos_fk_vis.astype(np.float32),
        'bvh_rotations': bvh.astype(np.float32),
        'anim_positions': pos_anim.astype(np.float32),
        'blender_key_loc': key_loc.astype(np.float32),
        'blender_key_rot': key_rot.qs.astype(np.float32),
        'gltf_local_t': loc_t.astype(np.float32),
        'gltf_local_r': loc_r.astype(np.float32),
        'gltf_local_s': loc_s.astype(np.float32),
        'gltf_world_pos': world_pos.astype(np.float32),
        'canonical_to_world': S.astype(np.float32),
    }
    meta = {'asset': name, 'glb': os.path.basename(glb), 'joint_node_names': jn,
            'fps': 30.0, 'num_frames': int(T), 'num_joints': nj,
            'similarity_scale': s_s,
            'world_vs_fk_max': float(np.abs(world_pos - np.einsum('ab,tjb->tja', S[:3, :3], pos_anim)
                                            - S[:3, 3]).max()),
            'ric_vs_fk_max': float(np.abs(pos_ric - pos_fk_vis).max())}
    print(f'[motion] {name}: world-vs-FK {meta["world_vs_fk_max"]:.3g}  RIC-vs-FK {meta["ric_vs_fk_max"]:.3g}'
          f'  sim scale {s_s:.6f}')
    return tensors, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', nargs='*', default=['eagle', 'shark'])
    ap.add_argument('--skip-model', action='store_true')
    ap.add_argument('--skip-ode', action='store_true')
    ap.add_argument('--euler-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.skip_ode:
        make_ode_goldens()
    if not a.skip_model:
        for name in a.assets:
            run_asset(name, euler_only=a.euler_only)


if __name__ == '__main__':
    main()
