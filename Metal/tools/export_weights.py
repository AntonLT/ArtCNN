#!/usr/bin/env python3
"""Write the conv weights of an ArtCNN GLSL shader as raw float32.

export_weights.py <shader.glsl> <out.bin>

For each conv pass in order: weight[out][in][3][3] then bias[out]. Prints the layer
shapes as "out,in,relu,skip" lines, where skip means the pass input is conv2d + previous.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from weights import load  # noqa: E402

passes, weights = load(sys.argv[1])
with open(sys.argv[2], "wb") as f:
    for p, wb in zip(passes, weights):
        if wb is None:
            continue
        w, b = wb
        f.write(w.astype(np.float32).tobytes())
        f.write(b.astype(np.float32).tobytes())
        print(f"{w.shape[0]},{w.shape[1]},{int('ReLU' in p.desc)},{int(len(p.binds) > 1)}")
