#!/usr/bin/env python3
"""Generate Metal kernels for an ArtCNN shader with Winograd 16->16 passes on simdgroup_matrix.

Usage: winograd.py GLSL/ArtCNN_C4F16_DS.glsl Metal/ArtCNN_C4F16_DS.metal

Each pass with 16 input and 16 output channels becomes a Winograd convolution, F(4, 3)
across and F(2, 3) down, whose products run on the 8x8 matrix hardware: in the Winograd
domain every transform position k is a product M_k[tiles][16 out] = V_k[tiles][16 in] x
U_k[16 in][16 out]. Tiles stay in registers for the whole layer.

The first conv (1->16) is never stored: the passes that read its output recompute it from
a staged luma block, which costs far less than writing and reading 16 channels. The
skip-connection conv writes the final doubled luma itself, replacing depth-to-space.
"""

import re
import sys

import numpy as np

import translate
from weights import load

G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], np.float64)

THREADS = 128
# The skip-connection pass computes one pixel per thread on a 16x8 block.
BLOCK_W, BLOCK_H = 16, 8
STAGE_W, STAGE_H = BLOCK_W + 2, BLOCK_H + 2


def half(v):
    return f"{float(v):.9g}"


def emit_luma_stage(emit, origin, w, h):
    """Stage luma for a w x h block of first-conv outputs (plus a one-pixel border)."""
    emit(f"    threadgroup half luma[{h + 2}][{w + 2}];")
    emit(f"    for (uint i = tid; i < {(h + 2) * (w + 2)}; i += {THREADS}) {{")
    emit(f"        luma[i / {w + 2}][i % {w + 2}] = load(LUMA, {origin} - 1 + int2(i % {w + 2}, i / {w + 2})).x;")
    emit("    }")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")


def conv0_expr(w0, b0, g, y, x):
    """First conv, channels 4g..4g+3 at staged luma position (x, y), in the GLSL's order."""
    terms = [f"half4({', '.join(half(v) for v in b0[g * 4:g * 4 + 4])})"]
    for ky in range(3):
        for kx in range(3):
            terms.append(f"half4({', '.join(half(w0[g * 4 + i, 0, ky, kx]) for i in range(4))}) "
                         f"* luma[{y} + {ky}][{x} + {kx}]")
    return " + ".join(terms)


def emit_conv0_pixels(emit, w0, b0, origin, store, sw=None, sh=None):
    """For each staged pixel, compute all 16 first-conv channels once and hand group g to store(g, value)."""
    sw, sh = sw or STAGE_W, sh or STAGE_H
    emit(f"    for (uint r = tid; r < {sh * sw}; r += {THREADS}) {{")
    emit(f"        const uint sy = r / {sw}, sx = r % {sw};")
    emit(f"        const int2 px = {origin} + int2(sx, sy);")
    emit("        const bool inside = all(px >= 0) && px.x < int(LUMA.get_width()) && px.y < int(LUMA.get_height());")
    for g in range(4):
        emit(f"        {store(g, f'inside ? {conv0_expr(w0, b0, g, 'sy', 'sx')} : half4(0.0h)')}")
    emit("    }")


def is_16_to_16(p, wb):
    return wb is not None and wb[0].shape[:2] == (16, 16) and len(p.binds) == 1


# Hybrid tiles: F(4, 3) across and F(2, 3) down, 4x2 output pixels from a 6x4 window.
F43 = dict(
    BT=np.array([[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0],
                 [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]], np.float64),
    G=np.array([[1 / 4, 0, 0], [-1 / 6, -1 / 6, -1 / 6], [-1 / 6, 1 / 6, -1 / 6],
                [1 / 24, 1 / 12, 1 / 6], [1 / 24, -1 / 12, 1 / 6], [0, 0, 1]], np.float64),
    AT=np.array([[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]], np.float64),
)
F23 = dict(
    BT=np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], np.float64),
    G=G,
    AT=np.array([[1, 1, 1, 0], [0, 1, -1, -1]], np.float64),
)
HYB_TX, HYB_TY = 8, 4  # tiles per threadgroup: one row of 8 per SIMD group
HYB_W, HYB_H = HYB_TX * 4, HYB_TY * 2  # 32x8 output pixels
HYB_SW, HYB_SH = HYB_W + 2, HYB_H + 2


def combo(row, names):
    """Linear combination sum(row[i] * names[i]) as MSL, skipping zeros and unit multiplies."""
    terms = []
    for coef, name in zip(row, names):
        if coef == 0:
            continue
        if coef == 1:
            terms.append(f"+ {name}")
        elif coef == -1:
            terms.append(f"- {name}")
        else:
            terms.append(f"{'-' if coef < 0 else '+'} {float(abs(coef))!r}h * {name}")
    text = " ".join(terms)
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def hybrid_kernel(index, p, w, b, conv0=None):
    """16->16 layer as Winograd F(4, 3) across x F(2, 3) down, with every tile in registers.

    SIMD group sg owns tile row sg (8 tiles of 4x2 pixels). Lane l is row r = (q & 4) + (l / 2) % 4
    and column pair c = (q & 2) * 2 + (l % 2) * 2 of an 8x8 simdgroup matrix (q = l / 4; checked
    at startup by ArtCNNMetal), so it transforms tile r's 6x4 window for channels c, c+1, 8+c,
    9+c: exactly its elements of the two V_k matrices for each of the 24 positions k. The
    products land in the same lane, which folds them into the output transform.
    """
    u = np.einsum("ak,oikl,bl->oiab", F23["G"], w.astype(np.float64), F43["G"])  # [out, in, 4, 6]
    table = np.zeros((24, 16, 16), np.float64)
    for a in range(4):
        for c in range(6):
            table[a * 6 + c] = u[:, :, a, c].T
    src = "LUMA" if conv0 else p.binds[0]
    relu = "ReLU" in p.desc
    lines = []
    emit = lines.append
    emit(f"constant half winograd_u{index}[24 * 16 * 16] = {{{', '.join(half(x) for x in table.ravel())}}};")
    emit(f"constant half winograd_b{index}[16] = {{{', '.join(half(x) for x in b)}}};")
    emit("")
    emit(f"// {p.desc} (Winograd F(4, 3) x F(2, 3), tiles in registers, products on simdgroup_matrix)")
    in_type = "texture2d<half>" if conv0 else "texture2d_array<half>"
    emit(f"kernel void artcnn_pass{index}({in_type} {src} [[texture(0)]],")
    emit("    texture2d_array<half, access::write> out_image [[texture(1)]],")
    emit("    uint2 tgid [[threadgroup_position_in_grid]],")
    emit("    uint tid [[thread_index_in_threadgroup]],")
    emit("    ushort sg [[simdgroup_index_in_threadgroup]],")
    emit("    ushort lane [[thread_index_in_simdgroup]])")
    emit("{")
    emit(f"    threadgroup half4 staged[4][{HYB_SH}][2][{HYB_SW // 2}];")
    emit(f"    const int2 block = int2(tgid) * int2({HYB_W}, {HYB_H});")
    if conv0:
        emit_luma_stage(emit, "block - 1", HYB_SW, HYB_SH)
        emit_conv0_pixels(emit, *conv0, "block - 1",
                          lambda g, v: f"staged[{g}][sy][sx & 1][sx >> 1] = {v};", HYB_SW, HYB_SH)
    else:
        emit(f"    for (uint i = tid; i < {4 * HYB_SH * HYB_SW}; i += {THREADS}) {{")
        emit(f"        uint g = i / {HYB_SH * HYB_SW}, r = i % {HYB_SH * HYB_SW};")
        emit(f"        int2 px = block - 1 + int2(r % {HYB_SW}, r / {HYB_SW});")
        emit(f"        staged[g][r / {HYB_SW}][(r % {HYB_SW}) & 1][(r % {HYB_SW}) >> 1] = "
             f"load({src}, px * 2 + int2(g & 1, g >> 1));")
        emit("    }")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    emit("")
    emit("    const uint q = lane / 4;")
    emit("    const uint mr = (q & 4) + (lane / 2) % 4, mc = (q & 2) * 2 + (lane % 2) * 2;")
    emit("    const uint tx = mr, ty = sg;")
    emit("    const uint g0 = mc / 4;")
    emit("    const bool hi = (mc & 2) != 0;")
    emit("    // Window rows r (4) and columns c (6) for channels mc, mc+1, 8+mc, 9+mc.")
    for r in range(4):
        for c in range(6):
            col = f"tx * 4 + {c}"
            emit(f"    half4 d{r}{c};")
            emit("    {")
            emit(f"        const half4 lo4 = staged[g0][ty * 2 + {r}][{c & 1}][tx * 2 + {c >> 1}];")
            emit(f"        const half4 hi4 = staged[g0 + 2][ty * 2 + {r}][{c & 1}][tx * 2 + {c >> 1}];")
            emit(f"        d{r}{c} = hi ? half4(lo4.zw, hi4.zw) : half4(lo4.xy, hi4.xy);")
            emit("    }")
    emit("    // Across: t = d B_x (F(4, 3)), then down: V = B_y^T t (F(2, 3)).")
    for r in range(4):
        for c in range(6):
            emit(f"    const half4 t{r}{c} = {combo(F43['BT'][c], [f'd{r}{cc}' for cc in range(6)])};")
    for a in range(4):
        for c in range(6):
            emit(f"    const half4 v{a}{c} = {combo(F23['BT'][a], [f't{r}{c}' for r in range(4)])};")
    emit("    half4 s[2][6];  // rows of A_y^T m")
    emit("    for (uint c = 0; c < 6; c++) {")
    emit("        s[0][c] = half4(0.0h);")
    emit("        s[1][c] = half4(0.0h);")
    emit("    }")
    for a in range(4):
        for c in range(6):
            k = a * 6 + c
            emit("    {")
            emit(f"        constant half *uk = winograd_u{index} + {k * 256};")
            emit("        simdgroup_half8x8 a0, a1, u00, u01, u10, u11;")
            emit(f"        a0.thread_elements()[0] = v{a}{c}.x;")
            emit(f"        a0.thread_elements()[1] = v{a}{c}.y;")
            emit(f"        a1.thread_elements()[0] = v{a}{c}.z;")
            emit(f"        a1.thread_elements()[1] = v{a}{c}.w;")
            for kh in range(2):
                for oh in range(2):
                    for e in range(2):
                        emit(f"        u{kh}{oh}.thread_elements()[{e}] = uk[({kh * 8} + mr) * 16 + {oh * 8} + mc + {e}];")
            emit("        simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h), c1 = c0;")
            emit("        simdgroup_multiply_accumulate(c0, a0, u00, c0);")
            emit("        simdgroup_multiply_accumulate(c0, a1, u10, c0);")
            emit("        simdgroup_multiply_accumulate(c1, a0, u01, c1);")
            emit("        simdgroup_multiply_accumulate(c1, a1, u11, c1);")
            emit("        const half4 m = half4(c0.thread_elements()[0], c0.thread_elements()[1], "
                 "c1.thread_elements()[0], c1.thread_elements()[1]);")
            for i in range(2):
                coef = F23["AT"][i][a]
                if coef == 1:
                    emit(f"        s[{i}][{c}] += m;")
                elif coef == -1:
                    emit(f"        s[{i}][{c}] -= m;")
            emit("    }")
    emit(f"    const half4 bias = half4(winograd_b{index}[mc], winograd_b{index}[mc + 1], "
         f"winograd_b{index}[8 + mc], winograd_b{index}[9 + mc]);")
    for i in range(2):
        for j in range(4):
            emit(f"    {{ // output row {i}, column {j}")
            emit(f"        half4 y = {combo(F43['AT'][j], [f's[{i}][{c}]' for c in range(6)])} + bias;")
            if relu:
                emit("        y = max(y, half4(0.0h));")
            emit("        const half4 other = simd_shuffle_xor(y, ushort(1));")
            emit("        const half4 even = hi ? other : y, odd = hi ? y : other;")
            emit("        const uint g = hi ? g0 + 2 : g0;")
            emit("        const half4 texel = hi ? half4(even.zw, odd.zw) : half4(even.xy, odd.xy);")
            emit(f"        store(out_image, (block + int2(tx * 4 + {j}, ty * 2 + {i})) * 2 + int2(g & 1, g >> 1), texel);")
            emit("    }")
    emit("}")
    return "\n".join(lines) + "\n"


def skip_kernel(index, p, w, b, conv0):
    """Skip-connection conv (16 -> 4) on conv0 + previous, writing the doubled luma."""
    assert w.shape[:2] == (4, 16) and p.binds[0] != "LUMA"
    prev = p.binds[1]
    lines = []
    emit = lines.append
    emit(f"// {p.desc} + depth-to-space, first conv recomputed from luma")
    emit(f"kernel void artcnn_pass{index}(texture2d<half> LUMA [[texture(0)]],")
    emit(f"    texture2d_array<half> {prev} [[texture(1)]],")
    emit("    texture2d<half, access::write> out_image [[texture(2)]],")
    emit("    uint2 tgid [[threadgroup_position_in_grid]],")
    emit("    uint tid [[thread_index_in_threadgroup]])")
    emit("{")
    emit(f"    threadgroup half4 staged[4][{STAGE_H}][{STAGE_W}];")
    emit(f"    const int2 block = int2(tgid) * int2({BLOCK_W}, {BLOCK_H});")
    emit_luma_stage(emit, "block - 1", STAGE_W, STAGE_H)
    emit_conv0_pixels(emit, *conv0, "block - 1",
                      lambda g, v: f"staged[{g}][sy][sx] = load({prev}, px * 2 + int2({g & 1}, {g >> 1})) + ({v});")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    emit(f"    const uint x = tid % {BLOCK_W}, y = tid / {BLOCK_W};")
    terms = [f"half4({', '.join(half(v) for v in b)})"]
    for ig in range(4):
        for ky in range(3):
            for kx in range(3):
                cols = ["half4(" + ", ".join(half(w[i, ig * 4 + j, ky, kx]) for i in range(4)) + ")" for j in range(4)]
                terms.append(f"half4x4({', '.join(cols)}) * staged[{ig}][y + {ky}][x + {kx}]")
    emit("    half4 result = " + "\n        + ".join(terms) + ";")
    emit("    // Depth-to-space: channel 2dy + dx goes to output pixel (2x + dx, 2y + dy).")
    emit("    const half4 v = clamp(result, 0.0h, 1.0h);")
    emit("    const int2 o = (block + int2(x, y)) * 2;")
    emit("    store(out_image, o + int2(0, 0), half4(v.x, 0.0h, 0.0h, 1.0h));")
    emit("    store(out_image, o + int2(1, 0), half4(v.y, 0.0h, 0.0h, 1.0h));")
    emit("    store(out_image, o + int2(0, 1), half4(v.z, 0.0h, 0.0h, 1.0h));")
    emit("    store(out_image, o + int2(1, 1), half4(v.w, 0.0h, 0.0h, 1.0h));")
    emit("}")
    return "\n".join(lines) + "\n"


def main(glsl_path, msl_path):
    translate.main(glsl_path, msl_path)
    text = open(msl_path).read()
    passes, weights = load(glsl_path)

    def kernel_of(index):
        m = re.search(rf"(?s)(?:// [^\n]*\n)?kernel void artcnn_pass{index}\(.*?\n}}\n", text)
        assert m, index
        return m.group(0)

    manifest = {index: line for line, index in re.findall(r"(?m)^(// PASS (\d+) .*)$", text)}
    first = passes[0]
    assert first.binds == ["LUMA"] and weights[0][0].shape[:2] == (16, 1)
    kernels, lines = [], []
    for index, (p, wb) in enumerate(zip(passes, weights)):
        if index == 0:
            continue  # the first conv is recomputed by the passes that read it
        if wb is None and p.binds == [passes[index - 1].save]:
            continue  # depth-to-space, written directly by the previous pass
        new = len(kernels)
        line = re.sub(r"^// PASS \d+", f"// PASS {new}", manifest[str(index)])
        binds = ",".join("LUMA" if name == first.save else name for name in p.binds)
        line = re.sub(r"binds=\S+", f"binds={binds}", line)
        reads_first = first.save in p.binds
        if is_16_to_16(p, wb):
            kernels.append(hybrid_kernel(new, p, *wb, conv0=weights[0] if reads_first else None))
            line = re.sub(r"block=\d+x\d+ threads=\d+x\d+", f"block={HYB_W * 2}x{HYB_H * 2} threads={THREADS}x1", line)
        elif reads_first and wb[0].shape[:2] == (4, 16):
            kernels.append(skip_kernel(new, p, *wb, weights[0]))
            line = re.sub(r"save=\S+", "save=OUTPUT", line)
            line = re.sub(r"block=\d+x\d+ threads=\d+x\d+", f"block={BLOCK_W}x{BLOCK_H} threads={THREADS}x1", line)
            line = re.sub(r"desc=(.*)$", rf"desc=\1 + {passes[index + 1].desc}", line)
        else:
            kernel = re.sub(rf"artcnn_pass{index}\(", f"artcnn_pass{new}(", kernel_of(index), count=1)
            following = passes[index + 1] if index + 1 < len(passes) else None
            if following is not None and weights[index + 1] is None and following.binds == [p.save]:
                # Depth-to-space: channel 2dy + dx goes to output pixel (2x + dx, 2y + dy).
                # Separate component writes; indexing result0 in a loop is 5x slower.
                old = "    store(out_image, output_base + int2(0, 0), result0);\n"
                assert old in kernel
                kernel = kernel.replace(old, (
                    "    const half4 y = clamp(result0, 0.0h, 1.0h);\n"
                    "    store(out_image, output_base * 2 + int2(0, 0), half4(y.x, 0.0h, 0.0h, 1.0h));\n"
                    "    store(out_image, output_base * 2 + int2(1, 0), half4(y.y, 0.0h, 0.0h, 1.0h));\n"
                    "    store(out_image, output_base * 2 + int2(0, 1), half4(y.z, 0.0h, 0.0h, 1.0h));\n"
                    "    store(out_image, output_base * 2 + int2(1, 1), half4(y.w, 0.0h, 0.0h, 1.0h));\n"))
                line = re.sub(r"save=\S+", "save=OUTPUT", line)
                line = re.sub(r"desc=(.*)$", rf"desc=\1 + {following.desc}", line)
            kernels.append(kernel)
        lines.append(line)

    prelude = text[:text.index("// PASS 0 ")]
    prelude = prelude.replace("Generated by Metal/translate.py", "Generated by Metal/winograd.py", 1)
    with open(msl_path, "w") as f:
        f.write(prelude + "\n".join(lines) + "\n\n" + "\n".join(kernels))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
