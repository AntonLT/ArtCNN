#!/usr/bin/env python3
"""Compare Winograd F(2x2, 3x3) and F(4x4, 3x3) accuracy in float16 for the ArtCNN 16->16 layers.

winograd_precision.py <shader.glsl>

Runs the network in float32, except the 16->16 convolutions, which run as Winograd in
float16 (transforms, products and sums rounded to float16 as on the GPU). Reports PSNR of
the final output against the float32 network.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from weights import load  # noqa: E402

F2 = dict(
    BT=np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], np.float64),
    G=np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], np.float64),
    AT=np.array([[1, 1, 1, 0], [0, 1, -1, -1]], np.float64),
)
F4 = dict(
    BT=np.array([[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0],
                 [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]], np.float64),
    G=np.array([[1 / 4, 0, 0], [-1 / 6, -1 / 6, -1 / 6], [-1 / 6, 1 / 6, -1 / 6],
                [1 / 24, 1 / 12, 1 / 6], [1 / 24, -1 / 12, 1 / 6], [0, 0, 1]], np.float64),
    AT=np.array([[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]], np.float64),
)


def conv3x3(x, w, b):
    c, h, wd = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((w.shape[0], h, wd), np.float32)
    for ky in range(3):
        for kx in range(3):
            out += np.tensordot(w[:, :, ky, kx], p[:, ky:ky + h, kx:kx + wd], axes=([1], [0]))
    return out + b[:, None, None]


def winograd16(x, w, b, m):
    """3x3 conv with zero padding as Winograd in float16; m = output tile size."""
    BT, G, AT = (m_[k] for m_, k in ((m, "BT"), (m, "G"), (m, "AT")))
    n = BT.shape[0]
    c, h, wd = x.shape
    th, tw = -(-h // (n - 2)), -(-wd // (n - 2))
    p = np.zeros((c, th * (n - 2) + 2, tw * (n - 2) + 2), np.float16)
    p[:, 1:h + 1, 1:wd + 1] = x
    # Tiles: [c, th, tw, n, n] with stride n - 2.
    idx_y = (np.arange(th) * (n - 2))[:, None] + np.arange(n)[None, :]
    idx_x = (np.arange(tw) * (n - 2))[:, None] + np.arange(n)[None, :]
    d = p[:, idx_y[:, None, :, None], idx_x[None, :, None, :]]
    bt = BT.astype(np.float16)
    v = np.einsum("ab,ctsbd->ctsad", bt, d).astype(np.float16)
    v = np.einsum("ctsad,ed->ctsae", v, bt).astype(np.float16)
    u = np.einsum("ak,oikl,bl->oiab", G, w.astype(np.float64), G).astype(np.float16)
    mm = np.einsum("oiab,itsab->otsab", u, v).astype(np.float16)
    at = AT.astype(np.float16)
    y = np.einsum("ra,otsab->otsrb", at, mm).astype(np.float16)
    y = np.einsum("otsrb,qb->otsrq", y, at).astype(np.float16)
    out = y.transpose(0, 1, 3, 2, 4).reshape(w.shape[0], th * (n - 2), tw * (n - 2))[:, :h, :wd]
    return out.astype(np.float32) + b[:, None, None]


def forward(passes, weights, x, m=None):
    feats = {"LUMA": x[None]}
    for p, wb in zip(passes, weights):
        if wb is None:
            y = feats[p.binds[0]]
            out = np.zeros((y.shape[1] * 2, y.shape[2] * 2), np.float32)
            for i in range(4):
                out[i // 2::2, i % 2::2] = y[i]
            return np.clip(out, 0.0, 1.0)
        inp = sum(feats[n] for n in p.binds)
        if m is not None and wb[0].shape[:2] == (16, 16) and len(p.binds) == 1:
            y = winograd16(inp.astype(np.float16), *wb, m)
        else:
            y = conv3x3(inp, *wb)
        if "ReLU" in p.desc:
            y = np.maximum(y, 0.0)
        feats[p.save] = y


def main(glsl):
    passes, weights = load(glsl)
    here = os.path.join(os.path.dirname(__file__), "..", "build")
    real = np.fromfile(os.path.join(here, "in.f32"), np.float32).reshape(1300, 1600)
    rng = np.random.default_rng(2)
    for name, x in (("real crop", real[300:812, 400:912]), ("noise", rng.random((256, 256), dtype=np.float32))):
        ref = forward(passes, weights, x)
        for label, m in (("F(2x2, 3x3)", F2), ("F(4x4, 3x3)", F4)):
            out = forward(passes, weights, x, m)
            mse = float(np.mean((out - ref) ** 2))
            codes = np.abs(np.round(out * 255) - np.round(ref * 255))
            print(f"{name:10s} {label}: PSNR {10 * np.log10(1 / mse):6.2f} dB vs float32, "
                  f"max {int(codes.max())} codes, {np.mean(codes > 0) * 100:5.2f}% of pixels differ")


if __name__ == "__main__":
    main(sys.argv[1])
