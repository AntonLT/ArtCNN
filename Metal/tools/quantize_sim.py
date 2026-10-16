#!/usr/bin/env python3
"""Simulate storing ArtCNN activations in 8 bits and measure the effect against ONNX.

quantize_sim.py <shader.glsl>

Calibrates a per-channel scale for every stored activation (max over a calibration set,
times a margin), quantizes the stored activations to 8 bits (unsigned after ReLU, signed
otherwise) in a NumPy forward pass, and compares the result with the unquantized forward
pass on held-out inputs. Prints the scales it would use.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from weights import load  # noqa: E402
from numpy_forward import conv3x3  # noqa: E402

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")


def forward(passes, weights, x, quant=None, record=None):
    """quant maps activation name -> (scale per channel, signed); record collects activations."""
    feats = {"LUMA": x[None]}
    for p, wb in zip(passes, weights):
        if wb is None:
            y = feats[p.binds[0]]
            out = np.zeros((y.shape[1] * 2, y.shape[2] * 2), np.float32)
            for i in range(4):
                out[i // 2::2, i % 2::2] = y[i]
            return np.clip(out, 0.0, 1.0)
        y = conv3x3(sum(feats[n] for n in p.binds), *wb)
        if "ReLU" in p.desc:
            y = np.maximum(y, 0.0)
        if record is not None:
            flat = np.abs(y).reshape(y.shape[0], -1)
            record.setdefault(p.save, []).append(np.stack([flat.max(axis=1), np.percentile(flat, 99.9, axis=1)]))
        if quant and p.save in quant:
            scale, signed, compand = quant[p.save]
            levels = 127.0 if signed else 255.0
            lo = -1.0 if signed else 0.0
            v = np.clip(y / scale[:, None, None], lo, 1.0)
            if compand:
                v = np.sign(v) * np.sqrt(np.abs(v))
            v = np.round(v * levels) / levels
            if compand:
                v = np.sign(v) * v * v
            y = v * scale[:, None, None]
        feats[p.save] = y


def inputs():
    rng = np.random.default_rng(1)
    real = np.fromfile(os.path.join(BUILD, "in.f32"), np.float32).reshape(1300, 1600)
    yy, xx = np.mgrid[0:256, 0:256]
    synthetic = {
        "noise": rng.random((256, 256), dtype=np.float32),
        "checker": (((yy // 1) + (xx // 1)) % 2).astype(np.float32),
        "edges": ((xx > 128) ^ (yy > 128)).astype(np.float32),
        "stripes": (xx % 4 < 2).astype(np.float32),
    }
    calib = {"real crop A": real[100:612, 200:712], **synthetic}
    held_out = {"real crop B": real[700:1212, 900:1412], "real full": real, "noise 2": rng.random((300, 300), dtype=np.float32)}
    return calib, held_out


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main(glsl):
    passes, weights = load(glsl)
    calib, held_out = inputs()
    record = {}
    for x in calib.values():
        forward(passes, weights, x, record=record)
    stored = [p for p in passes[1:-2]]  # activations written to memory between passes
    for label, stat, margin, compand in (("max x1.1, linear", 0, 1.1, False),
                                         ("max x1.1, sqrt companding", 0, 1.1, True),
                                         ("99.9th percentile, linear", 1, 1.0, False),
                                         ("99.9th percentile, sqrt companding", 1, 1.0, True)):
        quant = {}
        for p in stored:
            scale = np.max(np.stack(record[p.save])[:, stat], axis=0) * margin
            quant[p.save] = (np.maximum(scale, 1e-3), "ReLU" not in p.desc, compand)
        print(f"{label}:")
        for name, x in held_out.items():
            ref = forward(passes, weights, x)
            q = forward(passes, weights, x, quant=quant)
            codes = np.abs(np.round(ref * 255) - np.round(q * 255))
            print(f"  {name:12s} PSNR {psnr(ref, q):6.2f} dB vs unquantized, max {int(codes.max())} codes, "
                  f"{np.mean(codes > 0) * 100:5.2f}% of pixels differ")



if __name__ == "__main__":
    main(sys.argv[1])
