"""Golden tensors for the C++ T5 tokenizer / encoder (src/t5).

    .venv/Scripts/python.exe tools/python/t5_golden.py [--model models/flan-t5-base]

Runs the UPSTREAM ``T5Conditioner`` (reference/UniMate, unchanged; only its
``from_pretrained`` calls are redirected to the local model directory) and
writes into tests/golden/t5/:

  tokenizer.umtf  meta.strings (list), meta.n_required (the first n must match
                  exactly; the rest is a randomised non-ASCII stress set),
                  ids_flat i32, ids_offsets i32 [N+1] -- HF T5Tokenizer(text).input_ids
  encoder.umtf    meta.cases = [{name, texts}], and per case <name>.*:
                    input_ids i32 [B,T], attention_mask i32 [B,T] (after the
                    conditioner zeroes the rows of empty strings),
                    hidden f32 [B,T,768] (pool=False output),
                    pooled f32 [B,768]   (pool=True output),
                    embeddings f32 [B,T,768], block0 f32 [B,T,768] (output of
                    encoder.block[0]; only when B*T <= 512), position_bias f32 [1,12,T,T]
                  rel_buckets i32 [2*R+1] (torch bucket of relative position -R..R)
"""
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from umtf import write_umtf  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REF = os.path.join(ROOT, "reference", "UniMate")
OUT = os.path.join(ROOT, "tests", "golden", "t5")

LONG = ("A large four-legged creature slowly walks forward across uneven rocky ground, "
        "then suddenly turns to its left, raises both front legs high into the air, "
        "lets out a loud roar, and finally lands back down while its long tail sways "
        "from side to side in a wide, smooth arc before coming to a complete stop.")

EDGE = [
    "", " ", "   ", ".", "!?", "...", "Hello, world!", "a.b,c;d:e", "(left) [arm] {x} <y>",
    "~^`|\\@#$%&*_+=-/", "\"quoted\" 'single'", "0", "123", "3.14159", "1,000,000",
    "2nd 3rd 21st", "v2.0-beta", "UPPER lower MiXeD", "camelCaseJoint", "Left_Upper_Arm",
    "mixamorig:LeftHand", "Bip01 L Thigh", "  leading spaces", "trailing spaces   ",
    "multiple    internal   spaces", "\ttabs\tand\nnewlines\r\n", "a\r\nb", "x\x00y\x01z\x7f",
    # non-ASCII
    "Café déjà vu", "naïve façade", "Straße", "Ångström", "ﬁnal ﬂow", "ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３",
    "école", "ä́b", "日本語のテキスト", "Привет мир", "Γειά σου", "emoji 😀👍🏽",
    "flag 🇫🇷🇩🇪", "ZWJ 👨‍👩‍👧", "‘curly’ “quotes” — dash … ellipsis",
    "non breaking", "zero​width", "▁already▁meta", "▁▁x",
    "a ▁ b", "<extra_id_0> special </s> tokens <pad>", "<unk>", "x</s>y", "①②③ ½ ™ ©",
    "한국어 텍스트", "각", "नमस्ते", "क्षत्रिय", "　ideographic　space",
    "ǅ ǈ Ǌ", "µ °C ²³", "Ⅻ ⅷ", "﻿bom", "x́", "́lead",
]


def prompts():
    with open(os.path.join(REF, "assets", "examples", "examples.json"), encoding="utf-8") as f:
        return [e["prompt"] for e in json.load(f)]


def joint_labels():
    out = {}
    for a in ("eagle", "shark", "unitree_go2"):
        with open(os.path.join(REF, "assets", "examples", a, "annotation.json"), encoding="utf-8") as f:
            out[a] = [j["label"] for j in json.load(f)["joints"]]
    return out


def stress_strings(n, seed=1234):
    """Random strings biased towards characters the normaliser / grapheme
    clustering / unknown fusion treat specially."""
    rng = random.Random(seed)
    pools = [
        [chr(c) for c in range(0x20, 0x7F)],
        [chr(c) for c in range(0xA0, 0x250)],                 # Latin-1 + Latin ext
        [chr(c) for c in range(0x300, 0x370)],                # combining marks
        [chr(c) for c in range(0x370, 0x530)],                # Greek, Cyrillic
        [chr(c) for c in range(0x900, 0x980)],                # Devanagari
        [chr(c) for c in range(0x1100, 0x1200)],              # Hangul jamo
        [chr(c) for c in range(0xAC00, 0xAC40)],              # Hangul syllables
        [chr(c) for c in range(0x2000, 0x2070)],              # spaces, punctuation
        [chr(c) for c in range(0x2100, 0x2200)],              # letterlike, numbers
        [chr(c) for c in range(0x2460, 0x24FF)],              # enclosed
        [chr(c) for c in range(0x3000, 0x3100)],              # CJK punct, kana
        [chr(c) for c in range(0x4E00, 0x4E40)],              # CJK
        [chr(c) for c in range(0xFB00, 0xFB07)] + [chr(c) for c in range(0xFE00, 0xFE10)],
        [chr(c) for c in range(0xFF00, 0xFFF0)],              # fullwidth / halfwidth
        [chr(c) for c in range(0x1F1E6, 0x1F200)] + [chr(c) for c in range(0x1F300, 0x1F340)]
        + [chr(c) for c in range(0x1F3FB, 0x1F400)] + ["‍", "️"],
        [chr(c) for c in range(0x00, 0x20)] + ["\x7f", "\u0085", "­", "​", "▁",
                                               " ", "﻿", "�"],
        ["<pad>", "</s>", "<unk>", "<extra_id_1>", "<extra_id_99>", "▁"],
    ]
    out = []
    for _ in range(n):
        k = rng.randint(1, 12)
        s = []
        for _ in range(k):
            p = rng.choice(pools) if rng.random() < 0.6 else pools[0]
            s.append(rng.choice(p))
            if rng.random() < 0.25:
                s.append(" " * rng.randint(1, 3))
        out.append("".join(s))
    return out


def load_conditioner(model_dir, pool):
    sys.path.insert(0, REF)
    import unimate.models.text_encoder.t5 as t5mod
    from transformers import T5EncoderModel, T5Tokenizer

    class _Tok:
        @staticmethod
        def from_pretrained(name, **kw):
            assert name == "google/flan-t5-base", name
            return T5Tokenizer.from_pretrained(model_dir, **kw)

    class _Enc:
        @staticmethod
        def from_pretrained(name, **kw):
            assert name == "google/flan-t5-base", name
            return T5EncoderModel.from_pretrained(model_dir, **kw)

    t5mod.T5Tokenizer, t5mod.T5EncoderModel = _Tok, _Enc
    cond = t5mod.T5Conditioner("google/flan-t5-base", finetune=False, device="cpu",
                               autocast_dtype="float32", word_dropout=0.0,
                               normalize_text=False, pool=pool)
    return cond.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "flan-t5-base"))
    ap.add_argument("--n_stress", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    c_tok = load_conditioner(args.model, pool=False)
    c_pool = load_conditioner(args.model, pool=True)
    tok = c_tok.t5_tokenizer
    enc = c_tok.t5

    # ---------------------------------------------------------- tokenizer goldens
    joints = joint_labels()
    uniq_joints = list(dict.fromkeys(sum(joints.values(), [])))
    required = list(dict.fromkeys(prompts() + uniq_joints + EDGE + [LONG]))
    stress = [s for s in dict.fromkeys(stress_strings(args.n_stress)) if s not in required]
    strings = required + stress
    flat, offs = [], [0]
    for s in strings:
        ids = tok(s)["input_ids"]
        flat += ids
        offs.append(len(flat))
    write_umtf(os.path.join(OUT, "tokenizer.umtf"),
               {"ids_flat": np.asarray(flat, np.int32), "ids_offsets": np.asarray(offs, np.int32)},
               {"strings": strings, "n_required": len(required)})
    print(f"tokenizer.umtf: {len(required)} required + {len(stress)} stress strings")

    # ---------------------------------------------------------- encoder goldens
    cases = [(f"prompt{i}", [p]) for i, p in enumerate(prompts())]
    cases += [("edge", EDGE), ("long", [LONG, "", "Head"])]
    cases += [(f"joints_{a}", list(dict.fromkeys(v))) for a, v in joints.items()]
    joints72 = (joints["eagle"] + joints["unitree_go2"] + joints["shark"])[:72]
    cases += [("joints72", joints72)]
    cases += [("empty", [""])]

    captured = {}
    h1 = enc.encoder.embed_tokens.register_forward_hook(
        lambda m, i, o: captured.__setitem__("embeddings", o.detach().clone()))
    h2 = enc.encoder.block[0].register_forward_hook(
        lambda m, i, o: captured.__setitem__("block0", o[0].detach().clone()))

    tensors, meta_cases = {}, []
    for name, texts in cases:
        inputs = c_tok.tokenize(texts)
        hidden = c_tok(inputs)
        pooled = c_pool(c_pool.tokenize(texts))
        T = inputs["input_ids"].shape[1]
        attn = enc.encoder.block[0].layer[0].SelfAttention
        pb = attn.compute_bias(T, T)
        tensors[f"{name}.input_ids"] = inputs["input_ids"].numpy().astype(np.int32)
        tensors[f"{name}.attention_mask"] = inputs["attention_mask"].numpy().astype(np.int32)
        tensors[f"{name}.hidden"] = hidden.numpy()
        tensors[f"{name}.pooled"] = pooled.numpy()
        if len(texts) * T <= 512:  # keep the file small: intermediates for small cases
            tensors[f"{name}.embeddings"] = captured["embeddings"].numpy()
            tensors[f"{name}.block0"] = captured["block0"].numpy()
        tensors[f"{name}.position_bias"] = pb.numpy()
        meta_cases.append({"name": name, "texts": texts})
        print(f"  {name}: B={len(texts)} T={T}")
    h1.remove()
    h2.remove()

    R = 1024
    from transformers.models.t5.modeling_t5 import T5Attention
    rel = torch.arange(-R, R + 1)
    tensors["rel_buckets"] = T5Attention._relative_position_bucket(
        rel, bidirectional=True, num_buckets=32, max_distance=128).numpy().astype(np.int32)

    import transformers
    write_umtf(os.path.join(OUT, "encoder.umtf"), tensors,
               {"cases": meta_cases, "rel_range": R,
                "source": f"upstream T5Conditioner, transformers {transformers.__version__}, "
                          f"torch {torch.__version__}, attn {enc.config._attn_implementation}"})
    print(f"encoder.umtf: {len(cases)} cases")


if __name__ == "__main__":
    main()
