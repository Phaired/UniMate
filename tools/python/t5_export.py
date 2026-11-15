"""Export the flan-t5-base ENCODER + tokenizer tables to a UMTF file.

    .venv/Scripts/python.exe tools/python/t5_export.py \
        [--model models/flan-t5-base] [--out models/converted/flan-t5-base.umtf]

What the C++ side (src/t5) reads from the file
----------------------------------------------
Weights: the T5EncoderModel state-dict names, float32 (the tied duplicate
``encoder.embed_tokens.weight`` is dropped; ``shared.weight`` is the table).

Tokenizer -- everything the HF ``T5Tokenizer`` (transformers 5.x, i.e. the
``tokenizers`` Rust pipeline) does, without protobuf / sentencepiece:

  tokenizer.pieces          u8  [nbytes]  concatenated UTF-8 pieces (id order)
  tokenizer.piece_offsets   i32 [V+1]     byte offsets into ``pieces``
  tokenizer.scores          f32 [V]       unigram log-probs (exactly float32)
  tokenizer.piece_types     i32 [V]       1 normal, 2 unknown, 3 control,
                                          4 user-defined / added special token
  tokenizer.added_ids       i32 [A]       ids of the added (special) tokens,
                                          split out of the raw text first
  tokenizer.precompiled_charsmap u8 [n]   sentencepiece nmt_nfkc charsmap
                                          (u32 trie size | darts units | strings)
  tokenizer.gcb_ranges      i32 [R,3]     (first, last, props) code point ranges:
                                          props & 15 = grapheme-cluster-break class
                                          (see GCB_CLASSES), bit 4 Extended_Pictographic,
                                          bits 5-6 InCB (1 Linker, 2 Consonant, 3 Extend)

``meta`` holds the model config and the tokenizer settings.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from umtf import write_umtf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GCB_CLASSES = ["Other", "CR", "LF", "Control", "Extend", "ZWJ", "Regional_Indicator",
               "Prepend", "SpacingMark", "L", "V", "T", "LV", "LVT"]
INCB = {"Linker": 1, "Consonant": 2, "Extend": 3}


def _ranges(prop_regex, all_chars, regex):
    out = []
    for m in regex.finditer(prop_regex + "+", all_chars):
        out.append((ord(m.group(0)[0]), ord(m.group(0)[-1])))
    return out


def grapheme_table():
    """Per code point grapheme-cluster properties as compressed ranges."""
    import regex
    cps = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    all_chars = "".join(map(chr, cps))
    props = np.zeros(0x110000, dtype=np.int32)
    for cls_id, name in enumerate(GCB_CLASSES):
        if cls_id == 0:
            continue
        for a, b in _ranges(r"\p{Grapheme_Cluster_Break=%s}" % name, all_chars, regex):
            props[a:b + 1] = (props[a:b + 1] & ~15) | cls_id
    for a, b in _ranges(r"\p{Extended_Pictographic}", all_chars, regex):
        props[a:b + 1] |= 16
    for name, v in INCB.items():
        for a, b in _ranges(r"\p{InCB=%s}" % name, all_chars, regex):
            props[a:b + 1] |= v << 5
    rows, start = [], 0
    for c in range(1, 0x110001):
        if c == 0x110000 or props[c] != props[start]:
            if props[start] != 0:
                rows.append((start, c - 1, int(props[start])))
            start = c
    return np.asarray(rows, dtype=np.int32).reshape(-1, 3), regex.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "flan-t5-base"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "converted", "flan-t5-base.umtf"))
    args = ap.parse_args()

    import torch
    import sentencepiece as spm
    from transformers import T5EncoderModel, T5Tokenizer

    # ---------------------------------------------------------------- tokenizer
    tok = T5Tokenizer.from_pretrained(args.model)
    backend = json.loads(tok._tokenizer.to_str())  # the pipeline actually used
    model = backend["model"]
    assert model["type"] == "Unigram", model["type"]
    assert not model.get("byte_fallback"), "byte_fallback not supported"
    norm = backend["normalizer"]
    if norm["type"] == "Sequence":
        kinds = [n["type"] for n in norm["normalizers"]]
        assert kinds == ["Precompiled"], f"unexpected normalizer chain {kinds}"
        norm = norm["normalizers"][0]
    assert norm["type"] == "Precompiled", norm["type"]
    charsmap = base64.b64decode(norm["precompiled_charsmap"])
    pre = backend["pre_tokenizer"]
    assert pre["type"] == "Sequence" and [p["type"] for p in pre["pretokenizers"]] == \
        ["WhitespaceSplit", "Metaspace"], pre
    meta_sp = pre["pretokenizers"][1]
    assert meta_sp["replacement"] == "▁" and meta_sp["prepend_scheme"] == "always" \
        and meta_sp["split"], meta_sp
    post = backend["post_processor"]
    assert post["type"] == "TemplateProcessing" and \
        [list(x)[0] for x in post["single"]] == ["Sequence", "SpecialToken"], post

    vocab = model["vocab"]
    pieces = [p for p, _ in vocab]
    scores = np.asarray([s for _, s in vocab], dtype=np.float64)
    assert np.all(scores.astype(np.float32).astype(np.float64) == scores), "scores not float32"

    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.model, "spiece.model"))
    types = np.full(len(pieces), 4, dtype=np.int32)  # beyond spm: added special tokens
    for i in range(sp.get_piece_size()):
        assert sp.id_to_piece(i) == pieces[i]
        types[i] = 3 if sp.is_control(i) else 2 if sp.is_unknown(i) else 1
    added = [a for a in backend["added_tokens"]]
    for a in added:
        assert a["special"] and not a["normalized"] and not a["lstrip"] and not a["rstrip"] \
            and not a["single_word"], a
        assert pieces[a["id"]] == a["content"], a
    added_ids = np.asarray([a["id"] for a in added], dtype=np.int32)

    blob = b"".join(p.encode("utf-8") for p in pieces)
    offs = np.zeros(len(pieces) + 1, dtype=np.int32)
    offs[1:] = np.cumsum([len(p.encode("utf-8")) for p in pieces])
    gcb, regex_version = grapheme_table()

    import tokenizers, transformers
    tok_meta = {
        "type": "unigram",
        "vocab_size": len(pieces),
        "unk_id": int(model["unk_id"]),
        "pad_id": int(tok.pad_token_id),
        "eos_id": int(tok.eos_token_id),
        "min_score": float(scores.min()),
        "unk_penalty": 10.0,           # tokenizers::models::unigram K_UNK_PENALTY
        "fuse_unk": True,
        "normalizer": "precompiled(nmt_nfkc) per grapheme cluster, as tokenizers' Precompiled",
        "pre_tokenizer": "WhitespaceSplit + Metaspace(replacement=U+2581, prepend_scheme=always, split=true)",
        "post_processor": "append </s>",
        "spm_normalizer_spec": "nmt_nfkc add_dummy_prefix remove_extra_whitespaces",
        "grapheme_classes": GCB_CLASSES,
        "unicode_tables_from": f"regex {regex_version}",
        "source": f"transformers {transformers.__version__}, tokenizers {tokenizers.__version__}",
    }

    # ---------------------------------------------------------------- encoder
    enc = T5EncoderModel.from_pretrained(args.model).eval()
    cfg = enc.config
    sd = enc.state_dict()
    tensors = {}
    for k, v in sd.items():
        if k == "encoder.embed_tokens.weight":
            assert torch.equal(v, sd["shared.weight"])
            continue
        tensors[k] = v.detach().float().cpu().numpy()
    model_meta = {
        "model": "google/flan-t5-base",
        "arch": "t5-encoder",
        "d_model": cfg.d_model, "d_kv": cfg.d_kv, "d_ff": cfg.d_ff,
        "num_layers": cfg.num_layers, "num_heads": cfg.num_heads,
        "vocab_size": cfg.vocab_size,
        "relative_attention_num_buckets": cfg.relative_attention_num_buckets,
        "relative_attention_max_distance": cfg.relative_attention_max_distance,
        "layer_norm_epsilon": cfg.layer_norm_epsilon,
        "feed_forward_proj": cfg.feed_forward_proj,
        "dense_act_fn": cfg.dense_act_fn,
        "is_gated_act": bool(cfg.is_gated_act),
        "pad_token_id": cfg.pad_token_id, "eos_token_id": cfg.eos_token_id,
    }
    assert cfg.dense_act_fn == "gelu_new" and cfg.is_gated_act

    tensors["tokenizer.pieces"] = np.frombuffer(blob, dtype=np.uint8)
    tensors["tokenizer.piece_offsets"] = offs
    tensors["tokenizer.scores"] = scores.astype(np.float32)
    tensors["tokenizer.piece_types"] = types
    tensors["tokenizer.added_ids"] = added_ids
    tensors["tokenizer.precompiled_charsmap"] = np.frombuffer(charsmap, dtype=np.uint8)
    tensors["tokenizer.gcb_ranges"] = gcb

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_umtf(args.out, tensors, {"format": "unimate-t5/1", "config": model_meta,
                                   "tokenizer": tok_meta})
    n_w = sum(v.size for k, v in tensors.items() if not k.startswith("tokenizer."))
    print(f"wrote {args.out}: {n_w / 1e6:.1f}M encoder params, {len(pieces)} pieces, "
          f"{len(gcb)} grapheme ranges, {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
