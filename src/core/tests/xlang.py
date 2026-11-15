"""Cross-language UMTF helper: `xlang.py write PATH` / `xlang.py check PATH`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools", "python"))
import numpy as np  # noqa: E402
from umtf import read_umtf, write_umtf  # noqa: E402


def sample():
    return {
        "w": (np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.5 - 3.0),
        "ids": np.array([[1, -2, 3], [400000, 5, -6]], dtype=np.int32),
        "big": np.array([2**40, -(2**41), 7], dtype=np.int64),
        "mask": np.array([True, False, True, True, False], dtype=np.bool_),
        "scalar": np.float32(3.25),
        "empty": np.zeros((0, 5), dtype=np.float32),
        "odd": np.linspace(-1, 1, 7, dtype=np.float32),
    }


META = {"name": "xlang", "text": "héllo 世界", "n": 3, "nested": {"a": [1, 2.5, None]}}


def main():
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "write":
        write_umtf(path, sample(), META)
        return
    tensors, meta = read_umtf(path)
    exp = sample()
    assert meta == META, meta
    assert set(tensors) == set(exp), set(tensors)
    for k, v in exp.items():
        v = np.asarray(v)
        t = tensors[k]
        assert t.shape == v.shape, (k, t.shape, v.shape)
        if v.dtype == np.bool_:
            v = v.astype(np.uint8)
        assert np.array_equal(t, v), k
    print("python check ok")


main()
