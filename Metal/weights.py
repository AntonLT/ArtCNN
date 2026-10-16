"""Extract convolution weights from an ArtCNN mpv GLSL shader.

Each conv pass becomes (weight[out, in, ky, kx], bias[out]) in float32, read from the
literal constants of the translated passes. Channel c of a 16-channel tensor lives in
texel (c // 4) of the pass's 2x2 tile, component c % 4.
"""

import re

import numpy as np

from translate import parse


def conv_weights(p):
    m = re.search(r"shared (\w+) inp\[(\d+)\]", "\n".join(p.globals))
    elem, groups = m.group(1), int(m.group(2))
    in_channels = groups * (4 if elem == "V4" else 1)
    out_groups = len(re.findall(r"V4 result\d+ = ", p.body))
    w = np.zeros((out_groups * 4, in_channels, 3, 3), np.float32)
    b = np.zeros(out_groups * 4, np.float32)

    for r, values in re.findall(rf"V4 result(\d+) = V4\(([^)]*)\);", p.body):
        b[int(r) * 4:int(r) * 4 + 4] = [float(v) for v in values.split(",")]

    pattern = rf"result(\d+) \+= (M4|V4)\(([^)]*)\) \* inp_(\d+)_(\d+)_(\d+);"
    for r, kind, values, g, kx, ky in re.findall(pattern, p.body):
        r, g, kx, ky = int(r), int(g), int(kx), int(ky)
        v = [float(x) for x in values.split(",")]
        if kind == "V4":
            # Scalar input channel g times a vec4 of four output channels.
            w[r * 4:r * 4 + 4, g, ky, kx] += v
        else:
            # Column j of the mat4 multiplies input component j: out[i] += v[4j + i] * in[j].
            for j in range(4):
                for i in range(4):
                    w[r * 4 + i, g * 4 + j, ky, kx] += v[4 * j + i]
    return w, b


def load(glsl_path):
    passes = parse(open(glsl_path).read())
    return passes, [conv_weights(p) if "Depth-To-Space" not in p.desc else None for p in passes]
