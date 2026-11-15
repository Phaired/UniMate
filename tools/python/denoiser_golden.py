"""Golden tensors for the C++ denoiser (tests/golden/denoiser/*.umtf).

    .venv/Scripts/python.exe tools/python/denoiser_golden.py

Runs the REAL upstream model (EMA weights, CPU, float32) on REAL conditioning
built by the upstream sample.py path for ``--asset <dir> --prompt <text>``
(see denoiser_upstream.build_cond), for these cases:

    eagle   B=1  assets/examples/eagle  (51 joints)
    shark   B=1  assets/examples/shark  (16 joints)
    batch2  B=2  [eagle, shark] in one batch (different joint counts, padding)

Each file holds
    cond.<key>          every cond entry the model reads (upstream names/shapes,
                        batch dim included; integer keys stay int64)
    x                   fixed noise [B, J, D, F] (torch.randn, seed per case)
    t                   [T] flow times (float, passed to the model unscaled)
    out_cond.<i>        model(x, t_i, cond)                     [B, J, D, F]
    out_uncond.<i>      model(x, t_i, cond, force_mask=True)
    velocity.<i>        out_uncond + cfg * (out_cond - out_uncond)  (cfg in meta)
and, for t index ``meta.inter_t_index`` (conditional pass) unless --no-inter:
    inter.t_emb [B,D]  inter.caption_proj [B,D]  inter.tpos_emb [B,J,D]
    inter.tpos_pool [B,D]  inter.y [B,D]  inter.y_uncond [B,D]
    inter.input_layer [B,F+1,J,D]  inter.tokens [B,F+1,J,D]
    inter.rope_j.cos / .sin [B,J,head_dim]
    inter.graph_bias.<l> [B,H,J,J] (masked keys = -inf)
    inter.block0.spatial / .temporal / .ffn [B,F+1,J,D]  (residual stream)
    inter.last_block [B,F+1,J,D]
    inter.final [B,J,D_feat,F]  (== out_cond.<inter_t_index>)
The instrumented forward is asserted bit-identical to ``model(...)``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import denoiser_upstream as U  # noqa: E402
from umtf import write_umtf  # noqa: E402

EAGLE = ("assets/examples/eagle", "A bird flaps its wings and takes off.")
SHARK = ("assets/examples/shark", "A shark swims forward, swinging its tail.")
CASES = {
    "eagle": ([EAGLE], 1001, True),
    "shark": ([SHARK], 1002, True),
    "batch2": ([EAGLE, SHARK], 1003, False),
}
T_VALUES = [0.0, 0.3, 0.75, 1.0]
INTER_T = 1

# Keys of the cond dict UniMateGraphAdaLN (this config) reads.
MODEL_COND_KEYS = [
    "caption_emb", "n_joints", "motion_length", "tpos_first_frame",
    "tpos_first_frame_parents", "joint_depths", "joint_names_emb",
    "spectral_feats", "graph_dist", "joint_relations",
]


def instrumented_forward(model, x, ts, cond, force_mask):
    """graph_adaln.forward, step by step, collecting intermediates."""
    import torch
    import einops
    from unimate.models.denoiser.blocks.common import modulate
    from unimate.models.denoiser.blocks.graph import compute_graph_bias

    I = {}
    bs, nj, nf, nfr = x.shape
    temb = model.time_embedder(ts, dtype=x.dtype)
    capv = model._caption_vector(bs, cond, x.dtype, x.device, force_mask)
    cap = model.cond_embedder(capv)
    y = temb + cap
    I["t_emb"], I["caption_proj"] = temb, cap

    h, tpos_emb, jv = model.input_layer(x, cond)
    I["input_layer"], I["tpos_emb"] = h, tpos_emb
    tp = model.tpos_pool(tpos_emb, jv)
    y = y + tp
    I["tpos_pool"], I["y"] = tp, y

    h = model._apply_token_embeddings(h, cond)
    I["tokens"] = h

    tv = model.lengths_to_mask(cond["motion_length"] + model.n_prefix, nfr + model.n_prefix)
    temporal_mask = tv.unsqueeze(1).unsqueeze(1)
    joint_mask = jv.unsqueeze(1).unsqueeze(2)
    cos, sin = model.rope_j._compute_cos_sin(cond["spectral_feats"])
    I["rope_j.cos"], I["rope_j.sin"] = cos, sin
    tpos_ids = model.temporal_position_ids[:, :nfr + model.n_prefix]

    for li, block in enumerate(model.transformer_blocks):
        I[f"graph_bias.{li}"] = compute_graph_bias(block.s_attn, cond, joint_mask)
        if li == 0:
            (sh_s, sc_s, g_s, sh_t, sc_t, g_t, sh_m, sc_m, g_m) = \
                block.adaLN_modulation(y).chunk(9, dim=-1)
            e = lambda z: z[:, None, None, :]  # noqa: E731
            xs = modulate(block.norm_s(h), e(sh_s), e(sc_s))
            xs = block._forward_spatial(xs, joint_mask, None, cond, None)
            h = h + e(g_s) * xs
            I["block0.spatial"] = h
            xt = modulate(block.norm_t(h), e(sh_t), e(sc_t))
            xt = block._forward_temporal(xt, temporal_mask, tpos_ids)
            h = h + e(g_t) * xt
            I["block0.temporal"] = h
            xm = block.mlp(modulate(block.norm_mlp(h), e(sh_m), e(sc_m)))
            h = h + e(g_m) * xm
            I["block0.ffn"] = h
        else:
            h = block(h, y, spatial_mask=joint_mask, temporal_mask=temporal_mask,
                      position_ids_spatial=None, position_ids_temporal=tpos_ids,
                      cond=cond, graph_bias=None)
    I["last_block"] = h
    out = model.final_layer(h, y, joints_valid=jv)
    out = einops.rearrange(out, "b d f j -> b j d f")[:, :, :, model.n_prefix:]
    I["final"] = out
    return out, I


def make_case(model, config, name, cases, seed, with_inter):
    import torch
    cond = U.build_cond(config, cases, seed=seed)
    B = len(cases)
    shape = (B, config.dataset.max_joints, config.dataset.feature_len,
             config.dataset.max_motion_length)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    cfg = float(config.sampling.cfg_scale)

    tensors = {}
    for k in MODEL_COND_KEYS:
        v = cond[k]
        tensors[f"cond.{k}"] = v if v.is_floating_point() else v.to(torch.int64)
    tensors["x"] = x
    tensors["t"] = torch.tensor(T_VALUES, dtype=torch.float32)

    for i, tv in enumerate(T_VALUES):
        ts = torch.full((B,), tv, dtype=torch.float32)
        oc = model(x, ts, cond)
        ou = model(x, ts, cond, force_mask=True)
        tensors[f"out_cond.{i}"] = oc
        tensors[f"out_uncond.{i}"] = ou
        tensors[f"velocity.{i}"] = ou + cfg * (oc - ou)   # ClassifierFreeSampleModel
        if i == INTER_T and with_inter:
            out, inter = instrumented_forward(model, x, ts, cond, False)
            assert torch.equal(out, oc), "instrumented forward diverged from model()"
            _, inter_u = instrumented_forward(model, x, ts, cond, True)
            inter["y_uncond"] = inter_u["y"]
            for k, v in inter.items():
                tensors[f"inter.{k}"] = v

    meta = {
        "case": name,
        "model": U.MODEL_NAME,
        "assets": [a for a, _ in cases],
        "prompts": [p for _, p in cases],
        "object_type": cond["object_type"],
        "n_joints": [int(n) for n in cond["n_joints"]],
        "cfg_scale": cfg,
        "t_values": T_VALUES,
        "inter_t_index": INTER_T if with_inter else None,
        "noise_seed": seed,
        "cond_keys": MODEL_COND_KEYS,
        "torch_threads": torch.get_num_threads(),
    }
    os.makedirs(U.GOLDEN_DIR, exist_ok=True)
    path = os.path.join(U.GOLDEN_DIR, f"{name}.umtf")
    write_umtf(path, tensors, meta)
    o = tensors[f"out_cond.{INTER_T}"]
    print(f"wrote {path}: B={B} n_joints={meta['n_joints']} |out| max={float(o.abs().max()):.3f} "
          f"rms={float(o.pow(2).mean().sqrt()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--no-inter", action="store_true")
    args = ap.parse_args()
    model, config = U.load_model()
    for name in args.cases:
        cases, seed, inter = CASES[name]
        make_case(model, config, name, cases, seed, inter and not args.no_inter)


if __name__ == "__main__":
    main()
