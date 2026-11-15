"""Export the UniMate denoiser weights inference uses to UMTF.

    .venv/Scripts/python.exe tools/python/denoiser_export.py [--out PATH]

Builds the upstream model exactly like unimate/inference/sample.py
(``create_model`` + ``_load_checkpoint``: the checkpoint's ``model_state_dict``
first, then the EMA shadow parameters copied over every trainable parameter
when ``training.use_ema`` and ``ema_state_dict`` are present) and writes its
``state_dict()`` with the state-dict names as tensor names.

Besides the state dict, two derived tables are stored under ``derived.``:
  derived.rope_t.cos / derived.rope_t.sin  (max_motion_length + 1, head_dim)
      the temporal RopeND tables exactly as torch computes them (float32 pow /
      einsum / cos / sin), so the C++ side does not have to reproduce torch's
      transcendental rounding.

``meta`` = {"config": <config.json verbatim>, "derived": {...}, "source": {...}}.
Note the shared spectral RoPE module appears in the state dict both as
``rope_j.*`` and (aliased) as ``transformer_blocks.N.s_attn.rope.*``; both are
written, the C++ model reads ``rope_j.*``.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import denoiser_upstream as U  # noqa: E402
from umtf import write_umtf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=U.CONVERTED)
    args = ap.parse_args()

    import torch
    model, config = U.load_model()

    with open(os.path.join(U.MODEL_DIR, "config.json")) as f:
        config_json = json.load(f)

    # How far EMA moved the weights (sanity: proves the EMA copy happened).
    raw = torch.load(U.CKPT, map_location="cpu")
    raw_sd = raw.get("model_state_dict", raw)
    sd = model.state_dict()
    ema_delta = max(float((sd[k].float() - raw_sd[k].float()).abs().max())
                    for k in sd if k in raw_sd and sd[k].is_floating_point())
    uses_ema = bool(config.training.use_ema and "ema_state_dict" in raw)
    del raw, raw_sd

    tensors = {k: v.detach().float().contiguous() for k, v in sd.items()
               if v.is_floating_point()}

    # Temporal RoPE tables (RopeND, nd=1) exactly as the model builds them.
    nfr = model.max_motion_length + model.n_prefix
    pos = model.temporal_position_ids[:, :nfr]
    cos_parts, sin_parts = model.rope_t._get_pos_embs(pos, device=torch.device("cpu"))
    tensors["derived.rope_t.cos"] = cos_parts[0].float()
    tensors["derived.rope_t.sin"] = sin_parts[0].float()

    blk = model.transformer_blocks[0]
    derived = {
        "variant": f"{config.model.attention}+{config.model.text_cond}",
        "class": type(model).__name__,
        "text_dim": model.text_dim,
        "latent_dim": model.latent_dim,
        "num_heads": model.num_heads,
        "head_dim": model.latent_dim // model.num_heads,
        "num_layers": model.num_layers,
        "mlp_hidden": blk.mlp.w3.in_features,
        "feature_len": model.feature_len,
        "max_joints": model.max_joints,
        "max_motion_length": model.max_motion_length,
        "n_prefix": model.n_prefix,
        "max_frames_with_prefix": nfr,
        "max_depth": model.max_depth if model.use_depth_emb else 0,
        "timestep_frequency_embedding_size": model.time_embedder.frequency_embedding_size,
        "timestep_max_period": 10000,
        "timestep_input": "flow time t in [0, 1] as float, unscaled (transport.get_drift)",
        "rope_t_base": model.rope_t.bases[0],
        "rope_t_split_dims": model.rope_t.split_dims,
        "spectral_num_eigvecs": model.rope_j.num_eigvecs,
        "spectral_use_signnet": model.rope_j.use_signnet,
        "signnet_hidden": model.rope_j.spectral_encoder.phi[0].out_features,
        "graph_max_path_len": blk.s_attn.graph_dist_embedding.num_embeddings - 1,
        "graph_edge_types": blk.s_attn.graph_rel_embedding.num_embeddings,
        "tpos_pool_queries": model.num_tpos_queries,
        "tpos_pool_heads": model.tpos_pool.pool.cross_attn.num_heads,
        "tpos_pool_ffn_hidden": model.tpos_pool.pool.ffn.w3.in_features,
        "final_root_attn_heads": model.final_layer.root_cross_attn.num_heads,
        "rms_eps": 1e-6,
        "cond_mask_prob": model.cond_mask_prob,
        "weights": "ema" if uses_ema else "raw",
        "ema_max_abs_delta_vs_raw": ema_delta,
    }
    meta = {
        "format": "unimate-denoiser/1",
        "model_name": U.MODEL_NAME,
        "config": config_json,
        "derived": derived,
        "source": {"checkpoint": os.path.relpath(U.CKPT, U.ROOT).replace("\\", "/"),
                   "step": 150000},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_umtf(args.out, tensors, meta)
    n = sum(int(v.numel()) for v in tensors.values())
    print(f"wrote {args.out}: {len(tensors)} tensors, {n} floats, weights={derived['weights']} "
          f"(EMA max |delta| vs raw = {ema_delta:.3g})")


if __name__ == "__main__":
    main()
