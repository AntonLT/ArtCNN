#!/usr/bin/env python3
"""Generate Metal kernels for an ArtCNN shader with Winograd 16->16 passes on simdgroup_matrix.

Usage: winograd.py GLSL/ArtCNN_C4F16_DS.glsl Metal/ArtCNN_C4F16_DS.metal

Each pass with 16 input and 16 output channels becomes a Winograd F(2x2, 3x3) convolution
whose products run on the 8x8 matrix hardware: in the Winograd domain every transform
position k is a product M_k[tiles][16 out] = V_k[tiles][16 in] x U_k[16 in][16 out]. The
first conv stays as translated by translate.py, and the skip-connection conv writes the
final doubled luma itself, replacing the depth-to-space pass.
"""

import re
import sys

import numpy as np

import translate
from weights import load

G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], np.float64)

# Output block per threadgroup: 8x4 Winograd tiles (16x8 pixels), 128 threads. Each thread
# owns one tile and one group of 4 channels for the input and output transforms.
TILES_X, TILES_Y = 8, 4
TILES = TILES_X * TILES_Y
BLOCK_W, BLOCK_H = TILES_X * 2, TILES_Y * 2
THREADS = 128
STAGE_W, STAGE_H = BLOCK_W + 2, BLOCK_H + 2


def half(v):
    return f"{float(v):.9g}"


def transformed_weights(w):
    # U[o, i] = G g G^T, a 4x4 matrix per (output, input) channel pair.
    return np.einsum("ak,oikl,bl->oiab", G, w.astype(np.float64), G)


def is_16_to_16(p, wb):
    return wb is not None and wb[0].shape[:2] == (16, 16) and len(p.binds) == 1


def winograd_kernel(index, p, w, b):
    u = transformed_weights(w)  # [out, in, a, c]
    table = np.zeros((16, 16, 16), np.float64)  # [position][in][out]
    for k in range(16):
        a, c = divmod(k, 4)
        table[k] = u[:, :, a, c].T
    src = p.binds[0]
    relu = "ReLU" in p.desc
    lines = []
    emit = lines.append
    emit(f"constant half winograd_u{index}[16 * 16 * 16] = {{{', '.join(half(x) for x in table.ravel())}}};")
    emit(f"constant half winograd_b{index}[16] = {{{', '.join(half(x) for x in b)}}};")
    emit("")
    emit(f"// {p.desc} (Winograd F(2x2, 3x3) on simdgroup_matrix)")
    emit(f"kernel void artcnn_pass{index}(texture2d_array<half> {src} [[texture(0)]],")
    emit("    texture2d_array<half, access::write> out_image [[texture(1)]],")
    emit("    uint2 tgid [[threadgroup_position_in_grid]],")
    emit("    uint tid [[thread_index_in_threadgroup]],")
    emit("    ushort sg [[simdgroup_index_in_threadgroup]],")
    emit("    ushort lane [[thread_index_in_simdgroup]])")
    emit("{")
    # Threadgroup memory limits occupancy, so one 8 KB buffer holds first the staged input
    # (columns split by parity so lanes read consecutive addresses), then 8 transform
    # positions at a time.
    emit(f"    threadgroup half V[8][{TILES}][16];")
    emit(f"    threadgroup half4 (*staged)[{STAGE_H}][2][{STAGE_W // 2}] = "
         f"(threadgroup half4 (*)[{STAGE_H}][2][{STAGE_W // 2}])V;")
    emit(f"    const int2 block = int2(tgid) * int2({BLOCK_W}, {BLOCK_H});")
    emit(f"    for (uint i = tid; i < {4 * STAGE_H * STAGE_W}; i += {THREADS}) {{")
    emit(f"        uint g = i / {STAGE_H * STAGE_W}, r = i % {STAGE_H * STAGE_W};")
    emit(f"        int2 px = block - 1 + int2(r % {STAGE_W}, r / {STAGE_W});")
    emit(f"        staged[g][r / {STAGE_W}][(r % {STAGE_W}) & 1][(r % {STAGE_W}) >> 1] = "
         f"load({src}, px * 2 + int2(g & 1, g >> 1));")
    emit("    }")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    emit("")
    emit(f"    const uint tile = tid % {TILES}, grp = tid / {TILES};")
    emit(f"    const uint tx = tile % {TILES_X}, ty = tile / {TILES_X};")
    emit("    half4 t[4][4];  // B^T d for this thread's tile and input group")
    emit("    {")
    emit("        half4 d[4][4];")
    emit("        for (uint r = 0; r < 4; r++) {")
    emit("            for (uint c = 0; c < 4; c++) {")
    emit("                d[r][c] = staged[grp][ty * 2 + r][c & 1][tx + (c >> 1)];")
    emit("            }")
    emit("        }")
    emit("        for (uint c = 0; c < 4; c++) {")
    emit("            t[0][c] = d[0][c] - d[2][c];")
    emit("            t[1][c] = d[1][c] + d[2][c];")
    emit("            t[2][c] = d[2][c] - d[1][c];")
    emit("            t[3][c] = d[1][c] - d[3][c];")
    emit("        }")
    emit("    }")
    emit("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    emit("")
    emit("    // Lane l holds elements (mr, mc) and (mr, mc + 1) of an 8x8 simdgroup matrix,")
    emit("    // measured on an M4 Pro; used to fill the weight tiles from the constant table.")
    emit("    const uint q = lane / 4;")
    emit("    const uint mr = (q & 4) + (lane / 2) % 4, mc = (q & 2) * 2 + (lane % 2) * 2;")
    emit("    half4 s[2][4];  // rows of A^T m with A^T = [[1, 1, 1, 0], [0, 1, -1, -1]]")
    for batch in range(2):
        emit(f"    {{ // transform rows {batch * 2}-{batch * 2 + 1}: positions {batch * 8}-{batch * 8 + 7}")
        emit("        for (uint rr = 0; rr < 2; rr++) {")
        emit(f"            const uint r = {batch * 2} + rr;")
        emit("            const half4 v[4] = {t[r][0] - t[r][2], t[r][1] + t[r][2], t[r][2] - t[r][1], t[r][1] - t[r][3]};")
        emit("            for (uint c = 0; c < 4; c++) {")
        emit("                *(threadgroup half4 *)&V[rr * 4 + c][tile][grp * 4] = v[c];")
        emit("            }")
        emit("        }")
        emit("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        # Unrolled with constant names: indexing tile arrays with loop variables spills them.
        for slot in range(2):
            emit(f"        {{ // SIMD group sg multiplies local position sg * 2 + {slot}")
            emit(f"            const uint lk = sg * 2 + {slot};")
            emit(f"            constant half *uk = winograd_u{index} + ({batch * 8} + lk) * 256;")
            for kh in range(2):
                for oh in range(2):
                    emit(f"            simdgroup_half8x8 u{kh}{oh};")
                    for e in range(2):
                        emit(f"            u{kh}{oh}.thread_elements()[{e}] = uk[({kh * 8} + mr) * 16 + {oh * 8} + mc + {e}];")
            for tb in range(TILES // 8):
                emit("            {")
                emit(f"                threadgroup half *rows = &V[lk][{tb * 8}][0];")
                emit("                simdgroup_half8x8 a0, a1;")
                emit("                simdgroup_load(a0, rows, 16, ulong2(0, 0));")
                emit("                simdgroup_load(a1, rows, 16, ulong2(8, 0));")
                emit("                simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);")
                emit("                simdgroup_half8x8 c1 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);")
                emit("                simdgroup_multiply_accumulate(c0, a0, u00, c0);")
                emit("                simdgroup_multiply_accumulate(c0, a1, u10, c0);")
                emit("                simdgroup_multiply_accumulate(c1, a0, u01, c1);")
                emit("                simdgroup_multiply_accumulate(c1, a1, u11, c1);")
                emit("                simdgroup_store(c0, rows, 16, ulong2(0, 0));")
                emit("                simdgroup_store(c1, rows, 16, ulong2(8, 0));")
                emit("            }")
            emit("        }")
        emit("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        emit("        half4 m[8];")
        emit("        for (uint k = 0; k < 8; k++) {")
        emit("            m[k] = *(threadgroup half4 *)&V[k][tile][grp * 4];")
        emit("        }")
        emit("        for (uint c = 0; c < 4; c++) {")
        if batch == 0:
            emit("            s[0][c] = m[c] + m[4 + c];")
            emit("            s[1][c] = m[4 + c];")
        else:
            emit("            s[0][c] += m[c];")
            emit("            s[1][c] -= m[c] + m[4 + c];")
        emit("        }")
        if batch == 0:
            emit("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        emit("    }")
    emit(f"    const half4 bias = half4(winograd_b{index}[grp * 4], winograd_b{index}[grp * 4 + 1], "
         f"winograd_b{index}[grp * 4 + 2], winograd_b{index}[grp * 4 + 3]);")
    emit("    for (uint r = 0; r < 2; r++) {")
    emit("        half4 y0 = s[r][0] + s[r][1] + s[r][2] + bias;")
    emit("        half4 y1 = s[r][1] - s[r][2] - s[r][3] + bias;")
    if relu:
        emit("        y0 = max(y0, half4(0.0h));")
        emit("        y1 = max(y1, half4(0.0h));")
    emit("        const int2 sub = int2(grp & 1, grp >> 1);")
    emit("        store(out_image, (block + int2(tx * 2, ty * 2 + r)) * 2 + sub, y0);")
    emit("        store(out_image, (block + int2(tx * 2 + 1, ty * 2 + r)) * 2 + sub, y1);")
    emit("    }")
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
    kernels, lines = [], []
    for index, (p, wb) in enumerate(zip(passes, weights)):
        if wb is None and p.binds == [passes[index - 1].save]:
            continue  # depth-to-space, written directly by the previous pass
        new = len(kernels)
        line = re.sub(r"^// PASS \d+", f"// PASS {new}", manifest[str(index)])
        if is_16_to_16(p, wb):
            kernels.append(winograd_kernel(new, p, *wb))
            line = re.sub(r"block=\d+x\d+ threads=\d+x\d+", f"block={BLOCK_W * 2}x{BLOCK_H * 2} threads={THREADS}x1", line)
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
