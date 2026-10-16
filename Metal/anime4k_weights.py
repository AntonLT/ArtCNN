"""Parse an Anime4K v4 Restore CNN mpv shader into layers of convolution weights.

Each GLSL pass computes 4 output channels; passes that read the same textures form one layer.
A pass's inputs are its go_i(x, y) macros: a bound texture read with texOff (clamped at the
image edges, as libplacebo samples it), optionally through max(+-x, 0) (CReLU). Layer weights
are returned as w[out, in, ky, kx], with input channels ordered as go_0's 4, go_1's 4, ...
"""

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Layer:
    passes: list  # pass DESC strings, one per 4 outputs
    saves: list  # texture name written by each pass
    inputs: list  # per go_i: (texture, sign) with sign None (raw), +1 or -1 (max(+-x, 0))
    w: np.ndarray = None  # [out, in, 3, 3]
    b: np.ndarray = None  # [out]
    residual: bool = False  # result + MAIN
    out_channels: list = field(default_factory=list)  # valid outputs per pass


def parse_pass(chunk):
    headers, binds = {}, []
    lines = chunk.splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("//!"):
        key, _, value = lines[i][3:].partition(" ")
        if key == "BIND":
            binds.append(value.strip())
        else:
            headers[key] = value.strip()
        i += 1
    code = "\n".join(lines[i:])
    gos = {}
    for idx, body in re.findall(r"#define go_(\d+)\(x_off, y_off\) \((.*)\)\n", code):
        m = re.fullmatch(r"(\w+)_texOff\(vec2\(x_off, y_off\)\)", body)
        if m:
            gos[int(idx)] = (m.group(1), None)
            continue
        m = re.fullmatch(r"max\((-?)\((\w+)_texOff\(vec2\(x_off, y_off\)\)\), 0\.0\)", body)
        assert m, body
        gos[int(idx)] = (m.group(2), -1 if m.group(1) else 1)
    inputs = [gos[k] for k in sorted(gos)]
    w = np.zeros((4, 4 * len(inputs), 3, 3), np.float64)
    b = np.zeros(4, np.float64)
    for values, go, x, y in re.findall(r"mat4\(([^)]*)\) \* go_(\d+)\((-?[\d.]+), (-?[\d.]+)\)", code):
        v = [float(t) for t in values.split(",")]
        g, kx, ky = int(go), int(float(x)) + 1, int(float(y)) + 1
        # Column j multiplies input component j: out[o] += v[4j + o] * in[j].
        for j in range(4):
            for o in range(4):
                w[o, g * 4 + j, ky, kx] += v[4 * j + o]
    for values in re.findall(r"result \+= vec4\(([^)]*)\);", code):
        b += [float(t) for t in values.split(",")]
    residual = "return result + MAIN_tex(MAIN_pos);" in code
    return headers["DESC"], binds, headers["SAVE"], inputs, w, b, residual


def load(path):
    src = open(path).read()
    chunks = re.split(r"(?m)^(?=//!DESC )", src)[1:]
    layers = []
    for chunk in chunks:
        desc, binds, save, inputs, w, b, residual = parse_pass(chunk)
        if layers and layers[-1].inputs == inputs and not residual:
            layer = layers[-1]
            layer.passes.append(desc)
            layer.saves.append(save)
            layer.w = np.concatenate([layer.w, w])
            layer.b = np.concatenate([layer.b, b])
        else:
            layers.append(Layer([desc], [save], inputs, w, b, residual))
    for layer in layers:
        # A pass whose last output row and column are all zero only produces 3 channels (RGB).
        layer.out_channels = [3 if not np.any(layer.w[4 * k + 3]) and layer.b[4 * k + 3] == 0 else 4
                              for k in range(len(layer.passes))]
    return layers


def forward(layers, rgb):
    """NumPy reference: rgb is [H, W, 3] in [0, 1]; returns the restored [H, W, 3]."""
    h, wd = rgb.shape[:2]
    tex = {"MAIN": np.concatenate([rgb.transpose(2, 0, 1), np.ones((1, h, wd))]).astype(np.float64)}
    for layer in layers:
        parts = []
        for name, sign in layer.inputs:
            x = tex[name]
            parts.append(x if sign is None else np.maximum(sign * x, 0.0))
        x = np.concatenate(parts)
        p = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")  # texOff clamps at the edges
        y = np.zeros((layer.w.shape[0], h, wd))
        for ky in range(3):
            for kx in range(3):
                y += np.tensordot(layer.w[:, :, ky, kx], p[:, ky:ky + h, kx:kx + wd], axes=([1], [0]))
        y += layer.b[:, None, None]
        if layer.residual:
            return (y[:3] + tex["MAIN"][:3]).transpose(1, 2, 0)
        for k, name in enumerate(layer.saves):
            tex[name] = y[4 * k:4 * k + 4]
    raise ValueError("network has no residual output pass")
