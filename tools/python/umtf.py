"""UMTF ("UniMate Tensor File") reader / writer.

Python side of src/core/umtf.h -- see that header for the byte layout.
Only numpy is required.
"""
import json
import struct

import numpy as np

MAGIC = b"UMTF"
VERSION = 1
ALIGN = 64

_DTYPES = {"f32": np.dtype("<f4"), "i32": np.dtype("<i4"),
           "i64": np.dtype("<i8"), "u8": np.dtype("u1")}


def _dtype_name(arr):
    if arr.dtype == np.bool_ or arr.dtype == np.uint8:
        return "u8"
    if np.issubdtype(arr.dtype, np.floating):
        return "f32"
    if arr.dtype == np.int64:
        return "i64"
    if np.issubdtype(arr.dtype, np.integer):
        return "i32"
    raise TypeError(f"unsupported dtype {arr.dtype}")


def _pad(n):
    return (-n) % ALIGN


def write_umtf(path, tensors, meta=None):
    """Write ``tensors`` (dict name -> array-like / torch tensor) and ``meta``
    (JSON-serialisable dict) to ``path``. Floats are stored as float32."""
    entries, blobs, offset = [], [], 0
    for name, value in tensors.items():
        if hasattr(value, "detach"):  # torch.Tensor
            value = value.detach().cpu().float().numpy() if value.is_floating_point() \
                else value.detach().cpu().numpy()
        arr = np.asarray(value)
        dt = _dtype_name(arr)
        shape = arr.shape  # ascontiguousarray promotes 0-d arrays to 1-d
        arr = np.ascontiguousarray(arr.astype(_DTYPES[dt], copy=False)).reshape(shape)
        raw = arr.tobytes(order="C")
        entries.append({"name": name, "dtype": dt, "shape": list(arr.shape),
                        "offset": offset, "nbytes": len(raw)})
        blobs.append(raw)
        offset += len(raw) + _pad(len(raw))
    header = json.dumps({"meta": meta or {}, "tensors": entries},
                        ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IQ", VERSION, len(header)))
        f.write(header)
        f.write(b"\0" * _pad(16 + len(header)))
        for raw in blobs:
            f.write(raw)
            f.write(b"\0" * _pad(len(raw)))


def read_umtf(path):
    """Return ``(tensors, meta)`` with ``tensors`` a dict name -> np.ndarray."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path}: not a UMTF file")
        version, json_len = struct.unpack("<IQ", f.read(12))
        if version != VERSION:
            raise ValueError(f"{path}: unsupported UMTF version {version}")
        header = json.loads(f.read(json_len).decode("utf-8"))
        data_start = 16 + json_len + _pad(16 + json_len)
        tensors = {}
        for e in header["tensors"]:
            f.seek(data_start + e["offset"])
            arr = np.frombuffer(f.read(e["nbytes"]), dtype=_DTYPES[e["dtype"]])
            tensors[e["name"]] = arr.reshape(e["shape"]).copy()
    return tensors, header["meta"]
