#!/usr/bin/env python3
"""Prepare test inputs, run the official ONNX model and compare outputs.

prepare <image> <out.f32>          -> prints "W H"; BT.709 luma in [0, 1], like Inferencer
onnx <model.onnx> <in.f32> W H <out.f32>
compare <reference.f32> <candidate.f32> W H   (sizes of the doubled image)
"""

import sys

import numpy as np


def load(path, w, h):
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def prepare(image_path, out_path):
    from PIL import Image
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    luma = np.clip(luma, 0.0, 1.0).astype(np.float32)
    luma.tofile(out_path)
    print(luma.shape[1], luma.shape[0])


def run_onnx(model, in_path, w, h, out_path):
    import onnxruntime as ort
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    x = load(in_path, w, h)[None, None]
    y = session.run(None, {session.get_inputs()[0].name: x})[0]
    y[0, 0].astype(np.float32).tofile(out_path)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def compare(ref_path, cand_path, w, h):
    ref, cand = load(ref_path, w, h), load(cand_path, w, h)
    diff = np.abs(ref - cand)
    inner = diff[4:-4, 4:-4]
    print(f"max abs diff {diff.max():.5f} (interior {inner.max():.5f}), mean {diff.mean():.6f}, "
          f"PSNR {psnr(ref, cand):.2f} dB, 8-bit codes differing "
          f"{np.mean(np.round(ref * 255) != np.round(cand * 255)) * 100:.3f}% "
          f"(max {int(np.max(np.abs(np.round(ref * 255) - np.round(cand * 255))))} codes)")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "prepare":
        prepare(*args)
    elif cmd == "onnx":
        run_onnx(args[0], args[1], int(args[2]), int(args[3]), args[4])
    elif cmd == "compare":
        compare(args[0], args[1], int(args[2]), int(args[3]))
    else:
        sys.exit(f"unknown command {cmd}")
