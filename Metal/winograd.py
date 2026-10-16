#!/usr/bin/env python3
"""Generate Metal kernels for an ArtCNN shader with Winograd F(2x2, 3x3) 16->16 passes.

Usage: winograd.py GLSL/ArtCNN_C4F16_DS.glsl Metal/ArtCNN_C4F16_DS.metal [--fuse a,b ...]

Passes with 16 input and 16 output channels are rebuilt as Winograd convolutions using
weights extracted by weights.py. With --fuse a,b two consecutive 16->16 passes run in one
kernel that keeps the intermediate activations in threadgroup memory. The first conv, the
skip-connection conv and depth-to-space stay as translated by translate.py.
"""

import re
import sys

import numpy as np

import translate
from weights import load

G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], np.float64)

# Output block per threadgroup: 8x4 Winograd tiles (16x8 pixels), one SIMD group per
# output group of 4 channels.
TILES_X, TILES_Y = 8, 4
BLOCK_W, BLOCK_H = TILES_X * 2, TILES_Y * 2
THREADS = 128


def half(v):
    return f"{float(v):.9g}"


def transformed_weights(w):
    # U[o, i] = G g G^T, a 4x4 matrix per (output, input) channel pair.
    return np.einsum("ak,oikl,bl->oiab", G, w.astype(np.float64), G)


def emit_stage(emit, src, origin, sw, sh):
    """Load a [4][sh][2][sw / 2] block of 16-channel input with zero padding into `staged`.

    Columns are split by parity so the lanes of a Winograd read touch consecutive addresses.
    """
    emit(f"    for (uint i = tid; i < {4 * sh * sw}; i += {THREADS}) {{")
    emit(f"        uint g = i / {sh * sw}, r = i % {sh * sw};")
    emit(f"        int2 px = {origin} + int2(r % {sw}, r / {sw});")
    emit(f"        staged[g][r / {sw}][(r % {sw}) & 1][(r % {sw}) >> 1] = load({src}, px * 2 + int2(g & 1, g >> 1));")
    emit("    }")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")


def emit_winograd(emit, w, b, relu, src, tx, ty, indent):
    """Emit one Winograd tile: 2x2 output pixels of output group `sg` (4 channels).

    `src` is a threadgroup array [4][rows][2][cols / 2] of half4 whose element
    [g][2*ty+r][c & 1][tx + (c >> 1)] holds input pixel (c - 1, r - 1) of the tile. Leaves the results in `half4 y[2][2]`,
    indexed [row][col], with bias and optional ReLU applied.
    """
    u = transformed_weights(w)
    pad = " " * indent
    emit(f"{pad}half4 m[16];")
    emit(f"{pad}for (uint k = 0; k < 16; k++) {{")
    emit(f"{pad}    m[k] = half4(0.0h);")
    emit(f"{pad}}}")
    for ig in range(4):
        emit(f"{pad}{{ // input channels {ig * 4}-{ig * 4 + 3}: V = B^T d B")
        emit(f"{pad}    half4 d[4][4], t[4][4], V[16];")
        emit(f"{pad}    for (uint r = 0; r < 4; r++) {{")
        emit(f"{pad}        for (uint c = 0; c < 4; c++) {{")
        emit(f"{pad}            d[r][c] = {src}[{ig}][({ty}) * 2 + r][c & 1][({tx}) + (c >> 1)];")
        emit(f"{pad}        }}")
        emit(f"{pad}    }}")
        emit(f"{pad}    for (uint c = 0; c < 4; c++) {{")
        emit(f"{pad}        t[0][c] = d[0][c] - d[2][c];")
        emit(f"{pad}        t[1][c] = d[1][c] + d[2][c];")
        emit(f"{pad}        t[2][c] = d[2][c] - d[1][c];")
        emit(f"{pad}        t[3][c] = d[1][c] - d[3][c];")
        emit(f"{pad}    }}")
        emit(f"{pad}    for (uint r = 0; r < 4; r++) {{")
        emit(f"{pad}        V[r * 4 + 0] = t[r][0] - t[r][2];")
        emit(f"{pad}        V[r * 4 + 1] = t[r][1] + t[r][2];")
        emit(f"{pad}        V[r * 4 + 2] = t[r][2] - t[r][1];")
        emit(f"{pad}        V[r * 4 + 3] = t[r][1] - t[r][3];")
        emit(f"{pad}    }}")
        emit(f"{pad}    switch (sg) {{")
        for og in range(4):
            emit(f"{pad}    case {og}:")
            for k in range(16):
                a, c = divmod(k, 4)
                # out[i] += sum_j U[4og+i, 4ig+j, a, c] * V[j]; half4x4 columns are indexed by j.
                cols = ["half4(" + ", ".join(half(u[og * 4 + i, ig * 4 + j, a, c]) for i in range(4)) + ")"
                        for j in range(4)]
                emit(f"{pad}        m[{k}] += half4x4({', '.join(cols)}) * V[{k}];")
            emit(f"{pad}        break;")
        emit(f"{pad}    }}")
        emit(f"{pad}}}")
    emit(f"{pad}half4 bias;")
    emit(f"{pad}switch (sg) {{")
    for og in range(4):
        emit(f"{pad}case {og}: bias = half4(" + ", ".join(half(x) for x in b[og * 4:og * 4 + 4]) + "); break;")
    emit(f"{pad}}}")
    emit(f"{pad}// Y = A^T m A with A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]")
    emit(f"{pad}half4 y[2][2];")
    emit(f"{pad}for (uint r = 0; r < 2; r++) {{")
    emit(f"{pad}    half4 s[4];")
    emit(f"{pad}    for (uint c = 0; c < 4; c++) {{")
    emit(f"{pad}        s[c] = r == 0 ? m[0 * 4 + c] + m[1 * 4 + c] + m[2 * 4 + c]")
    emit(f"{pad}                      : m[1 * 4 + c] - m[2 * 4 + c] - m[3 * 4 + c];")
    emit(f"{pad}    }}")
    emit(f"{pad}    y[r][0] = s[0] + s[1] + s[2] + bias;")
    emit(f"{pad}    y[r][1] = s[1] - s[2] - s[3] + bias;")
    if relu:
        emit(f"{pad}    y[r][0] = max(y[r][0], half4(0.0h));")
        emit(f"{pad}    y[r][1] = max(y[r][1], half4(0.0h));")
    emit(f"{pad}}}")


def kernel_header(emit, index, binds):
    params = [f"texture2d<half> {name} [[texture({i})]]" for i, name in enumerate(binds)]
    params.append(f"texture2d<half, access::write> out_image [[texture({len(binds)})]]")
    params += ["uint2 tgid [[threadgroup_position_in_grid]]",
               "uint tid [[thread_index_in_threadgroup]]",
               "ushort sg [[simdgroup_index_in_threadgroup]]",
               "ushort lane [[thread_index_in_simdgroup]]"]
    emit(f"kernel void artcnn_pass{index}(" + ",\n    ".join(params) + ")")
    emit("{")


def emit_store(emit, base, indent):
    pad = " " * indent
    emit(f"{pad}for (uint r = 0; r < 2; r++) {{")
    emit(f"{pad}    for (uint c = 0; c < 2; c++) {{")
    emit(f"{pad}        store(out_image, ({base} + int2(c, r)) * 2 + int2(sg & 1, sg >> 1), y[r][c]);")
    emit(f"{pad}    }}")
    emit(f"{pad}}}")


def single_kernel(index, p, w, b):
    lines = []
    emit = lines.append
    emit(f"// {p.desc} (Winograd F(2x2, 3x3))")
    kernel_header(emit, index, p.binds)
    sw, sh = BLOCK_W + 2, BLOCK_H + 2
    emit(f"    threadgroup half4 staged[4][{sh}][2][{sw // 2}];")
    emit(f"    const int2 block = int2(tgid) * int2({BLOCK_W}, {BLOCK_H});")
    emit_stage(emit, p.binds[0], "block - 1", sw, sh)
    emit(f"    const uint tx = lane % {TILES_X}, ty = lane / {TILES_X};")
    emit("    {")
    emit_winograd(emit, w, b, "ReLU" in p.desc, "staged", "tx", "ty", 8)
    emit_store(emit, "block + int2(tx, ty) * 2", 8)
    emit("    }")
    emit("}")
    return "\n".join(lines) + "\n"


def fused_kernel(index, pa, wa, ba, pb, wb, bb):
    """Two consecutive 16->16 convs; the first layer's output stays in threadgroup memory."""
    mtx, mty = TILES_X + 1, TILES_Y + 1  # first layer: output block plus a one-pixel border
    mw, mh = mtx * 2, mty * 2
    sw, sh = mw + 2, mh + 2
    lines = []
    emit = lines.append
    emit(f"// {pa.desc} + {pb.desc} (fused Winograd F(2x2, 3x3))")
    kernel_header(emit, index, pa.binds)
    emit(f"    threadgroup half4 staged[4][{sh}][2][{sw // 2}];")
    emit(f"    threadgroup half4 mid[4][{mh}][2][{mw // 2}];")
    emit(f"    const int2 block = int2(tgid) * int2({BLOCK_W}, {BLOCK_H});")
    emit(f"    const int2 size = int2(out_image.get_width(), out_image.get_height()) / 2;")
    emit_stage(emit, pa.binds[0], "block - 2", sw, sh)
    emit(f"    for (uint tile = lane; tile < {mtx * mty}; tile += 32) {{")
    emit(f"        const uint tx = tile % {mtx}, ty = tile / {mtx};")
    emit_winograd(emit, wa, ba, "ReLU" in pa.desc, "staged", "tx", "ty", 8)
    emit("        for (uint r = 0; r < 2; r++) {")
    emit("            for (uint c = 0; c < 2; c++) {")
    emit("                // The second layer pads with zeros outside the image, not with this layer's output.")
    emit("                int2 px = block - 1 + int2(tx * 2 + c, ty * 2 + r);")
    emit("                bool inside = all(px >= 0) && all(px < size);")
    emit("                mid[sg][ty * 2 + r][c][tx] = inside ? y[r][c] : half4(0.0h);")
    emit("            }")
    emit("        }")
    emit("    }")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    emit(f"    const uint tx = lane % {TILES_X}, ty = lane / {TILES_X};")
    emit("    {")
    emit_winograd(emit, wb, bb, "ReLU" in pb.desc, "mid", "tx", "ty", 8)
    emit_store(emit, "block + int2(tx, ty) * 2", 8)
    emit("    }")
    emit("}")
    return "\n".join(lines) + "\n"


def is_16_to_16(p, wb):
    return wb is not None and wb[0].shape[:2] == (16, 16) and len(p.binds) == 1


def main(glsl_path, msl_path, fuse):
    translate.main(glsl_path, msl_path)
    text = open(msl_path).read()
    passes, weights = load(glsl_path)

    def kernel_of(index):
        m = re.search(rf"(?s)(?:// [^\n]*\n)?kernel void artcnn_pass{index}\(.*?\n}}\n", text)
        assert m, index
        return m.group(0)

    manifest = {index: line for line, index in re.findall(r"(?m)^(// PASS (\d+) .*)$", text)}
    fused_first = {a: b for a, b in fuse}
    fused_second = {b for _, b in fuse}

    kernels, lines = [], []
    for index, (p, wb) in enumerate(zip(passes, weights)):
        if index in fused_second:
            continue
        new = len(kernels)
        line = manifest[str(index)]
        if index in fused_first:
            b_index = fused_first[index]
            pb, wbb = passes[b_index], weights[b_index]
            assert b_index == index + 1 and is_16_to_16(p, wb) and is_16_to_16(pb, wbb)
            kernels.append(fused_kernel(new, p, *wb, pb, *wbb))
            line = re.sub(r"save=\S+", f"save={pb.save}", line)
            line = re.sub(r"desc=(.*)$", rf"desc=\1 + {pb.desc}", line)
        elif is_16_to_16(p, wb):
            kernels.append(single_kernel(new, p, *wb))
        else:
            kernels.append(re.sub(rf"artcnn_pass{index}\(", f"artcnn_pass{new}(", kernel_of(index), count=1))
            lines.append(re.sub(r"^// PASS \d+", f"// PASS {new}", line))
            continue
        line = re.sub(r"block=\d+x\d+ threads=\d+x\d+", f"block={BLOCK_W * 2}x{BLOCK_H * 2} threads={THREADS}x1", line)
        lines.append(re.sub(r"^// PASS \d+", f"// PASS {new}", line))

    prelude = text[:text.index("// PASS 0 ")]
    prelude = prelude.replace("Generated by Metal/translate.py", "Generated by Metal/winograd.py", 1)
    with open(msl_path, "w") as f:
        f.write(prelude + "\n".join(lines) + "\n\n" + "\n".join(kernels))


if __name__ == "__main__":
    fuse = [tuple(int(x) for x in arg.split(",")) for arg in sys.argv[3:] if arg != "--fuse"]
    main(sys.argv[1], sys.argv[2], fuse)
