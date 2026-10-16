#!/usr/bin/env python3
"""Run ArtCNN in NumPy from weights extracted out of the GLSL, to check the extraction.

numpy_forward.py <shader.glsl> <in.f32> W H <out.f32>
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from weights import load  # noqa: E402


def conv3x3(x, w, b):
    c, h, wd = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((w.shape[0], h, wd), np.float32)
    for ky in range(3):
        for kx in range(3):
            patch = padded[:, ky:ky + h, kx:kx + wd]
            out += np.tensordot(w[:, :, ky, kx], patch, axes=([1], [0]))
    return out + b[:, None, None]


def forward(glsl, x):
    passes, weights = load(glsl)
    feats = {"LUMA": x[None]}
    for p, wb in zip(passes, weights):
        if wb is None:
            y = feats[p.binds[0]]
            h, wd = y.shape[1:]
            out = np.zeros((h * 2, wd * 2), np.float32)
            for i in range(4):
                out[i // 2::2, i % 2::2] = y[i]
            return np.clip(out, 0.0, 1.0)
        inp = sum(feats[name] for name in p.binds)
        y = conv3x3(inp, *wb)
        if "ReLU" in p.desc:
            y = np.maximum(y, 0.0)
        feats[p.save] = y


if __name__ == "__main__":
    glsl, in_path, w, h, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    x = np.fromfile(in_path, np.float32).reshape(h, w)
    forward(glsl, x).astype(np.float32).tofile(out_path)
