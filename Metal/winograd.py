#!/usr/bin/env python3
"""Generate Metal kernels for an ArtCNN shader with Winograd F(2x2, 3x3) 16->16 passes.

Usage: winograd.py GLSL/ArtCNN_C4F16_DS.glsl Metal/ArtCNN_C4F16_DS.metal

Passes with 16 input and 16 output channels are rebuilt as Winograd convolutions using
weights extracted by weights.py. The first conv, the skip-connection conv and
depth-to-space stay as translated by translate.py.
"""

import re
import sys

import numpy as np

import translate
from weights import load

TILES_X, TILES_Y = 8, 4  # Winograd tiles per threadgroup; one SIMD group (32 lanes) per output group
BLOCK_W, BLOCK_H = TILES_X * 2, TILES_Y * 2  # output pixels per threadgroup

G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], np.float64)


def half(v):
    return f"{float(v):.9g}"


def transformed_weights(w):
    # U[o, i] = G g G^T, a 4x4 matrix per (output, input) channel pair.
    return np.einsum("ak,oikl,bl->oiab", G, w.astype(np.float64), G)


def winograd_kernel(index, p, w, b):
    assert w.shape[:2] == (16, 16), w.shape
    u = transformed_weights(w)
    src = p.binds[0]
    relu = "ReLU" in p.desc
    lines = []
    emit = lines.append
    emit(f"// {p.desc} (Winograd F(2x2, 3x3))")
    emit(f"kernel void artcnn_pass{index}(texture2d<half> {src} [[texture(0)]],")
    emit("    texture2d<half, access::write> out_image [[texture(1)]],")
    emit("    uint2 tgid [[threadgroup_position_in_grid]],")
    emit("    uint tid [[thread_index_in_threadgroup]],")
    emit("    ushort sg [[simdgroup_index_in_threadgroup]],")
    emit("    ushort lane [[thread_index_in_simdgroup]])")
    emit("{")
    sw, sh = BLOCK_W + 2, BLOCK_H + 2
    emit(f"    threadgroup half4 staged[4][{sh}][{sw}];")
    emit(f"    const int2 origin = int2(tgid) * int2({BLOCK_W}, {BLOCK_H}) - 1;")
    emit(f"    for (uint i = tid; i < {4 * sh * sw}; i += {4 * 32}) {{")
    emit(f"        uint g = i / {sh * sw}, r = i % {sh * sw};")
    emit(f"        int2 px = origin + int2(r % {sw}, r / {sw});")
    emit(f"        staged[g][r / {sw}][r % {sw}] = load({src}, px * 2 + int2(g & 1, g >> 1));")
    emit("    }")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    emit("")
    emit(f"    const uint tx = lane % {TILES_X}, ty = lane / {TILES_X};")
    emit("    half4 m[16];")
    emit("    for (uint k = 0; k < 16; k++) {")
    emit("        m[k] = half4(0.0h);")
    emit("    }")
    for ig in range(4):
        emit(f"    {{ // input channels {ig * 4}-{ig * 4 + 3}: V = B^T d B")
        emit("        half4 d[4][4], t[4][4], V[16];")
        emit("        for (uint r = 0; r < 4; r++) {")
        emit("            for (uint c = 0; c < 4; c++) {")
        emit(f"                d[r][c] = staged[{ig}][ty * 2 + r][tx * 2 + c];")
        emit("            }")
        emit("        }")
        emit("        for (uint c = 0; c < 4; c++) {")
        emit("            t[0][c] = d[0][c] - d[2][c];")
        emit("            t[1][c] = d[1][c] + d[2][c];")
        emit("            t[2][c] = d[2][c] - d[1][c];")
        emit("            t[3][c] = d[1][c] - d[3][c];")
        emit("        }")
        emit("        for (uint r = 0; r < 4; r++) {")
        emit("            V[r * 4 + 0] = t[r][0] - t[r][2];")
        emit("            V[r * 4 + 1] = t[r][1] + t[r][2];")
        emit("            V[r * 4 + 2] = t[r][2] - t[r][1];")
        emit("            V[r * 4 + 3] = t[r][1] - t[r][3];")
        emit("        }")
        emit("        switch (sg) {")
        for og in range(4):
            emit(f"        case {og}:")
            for k in range(16):
                a, c = divmod(k, 4)
                # out[i] += sum_j U[4og+i, 4ig+j, a, c] * V[j]; half4x4 columns are indexed by j.
                cols = ["half4(" + ", ".join(half(u[og * 4 + i, ig * 4 + j, a, c]) for i in range(4)) + ")"
                        for j in range(4)]
                emit(f"            m[{k}] += half4x4({', '.join(cols)}) * V[{k}];")
            emit("            break;")
        emit("        }")
        emit("    }")
    emit("    half4 bias;")
    emit("    switch (sg) {")
    for og in range(4):
        emit(f"    case {og}: bias = half4(" + ", ".join(half(x) for x in b[og * 4:og * 4 + 4]) + "); break;")
    emit("    }")
    emit("")
    emit("    // Y = A^T m A with A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]")
    emit("    half4 s[2][4];")
    emit("    for (uint c = 0; c < 4; c++) {")
    emit("        s[0][c] = m[0 * 4 + c] + m[1 * 4 + c] + m[2 * 4 + c];")
    emit("        s[1][c] = m[1 * 4 + c] - m[2 * 4 + c] - m[3 * 4 + c];")
    emit("    }")
    emit(f"    const int2 base = int2(tgid) * int2({BLOCK_W}, {BLOCK_H}) + int2(tx, ty) * 2;")
    emit("    const int2 sub = int2(sg & 1, sg >> 1);")
    emit("    for (uint r = 0; r < 2; r++) {")
    emit("        half4 y0 = s[r][0] + s[r][1] + s[r][2] + bias;")
    emit("        half4 y1 = s[r][1] - s[r][2] - s[r][3] + bias;")
    if relu:
        emit("        y0 = max(y0, half4(0.0h));")
        emit("        y1 = max(y1, half4(0.0h));")
    emit("        store(out_image, (base + int2(0, r)) * 2 + sub, y0);")
    emit("        store(out_image, (base + int2(1, r)) * 2 + sub, y1);")
    emit("    }")
    emit("}")
    return "\n".join(lines) + "\n"


def main(glsl_path, msl_path):
    translate.main(glsl_path, msl_path)
    text = open(msl_path).read()
    passes, weights = load(glsl_path)
    for index, (p, wb) in enumerate(zip(passes, weights)):
        if wb is None or wb[0].shape[:2] != (16, 16) or len(p.binds) != 1:
            continue
        kernel = re.search(rf"(?s)// [^\n]*\nkernel void artcnn_pass{index}\(.*?\n}}\n", text)
        assert kernel, index
        text = text.replace(kernel.group(0), winograd_kernel(index, p, *wb))
        text = re.sub(rf"(// PASS {index} .*?)block=\d+x\d+ threads=\d+x\d+",
                      rf"\1block={BLOCK_W * 2}x{BLOCK_H * 2} threads=128x1", text)
    text = text.replace("Generated by Metal/translate.py", "Generated by Metal/winograd.py", 1)
    with open(msl_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
