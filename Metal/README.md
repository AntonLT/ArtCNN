# ArtCNN on Metal

Native Metal (MSL 4.1) versions of the ArtCNN GLSL shaders, generated from the official
`GLSL/*.glsl` files so the weights stay identical. Only `C4F16_DS` is generated so far.

- `translate.py` converts each mpv/libplacebo pass into a Metal compute kernel, line for line.
- `winograd.py` builds on it and replaces the five 16->16 convolutions with Winograd
  F(2x2, 3x3) kernels whose products run on the 8x8 matrix hardware (`simdgroup_matrix`),
  using weights extracted by `weights.py`. The skip-connection conv writes the final doubled
  luma directly, so there is no separate depth-to-space pass. This is the shipped version.
- `ArtCNNMetal.h/.m` compiles a generated shader and encodes its passes into a command
  buffer. Input is a single-channel luma texture; output is an R16Float texture at 2x.

Regenerate:

```
python winograd.py ../GLSL/ArtCNN_C4F16_DS.glsl ArtCNN_C4F16_DS.metal
```

## Verification

`tools/` holds the checks. `reference.py` prepares BT.709 luma inputs and runs the official
ONNX model; `artcnn_metal_cli.m` runs a generated shader; `plbench.c` runs the original GLSL
through libplacebo on MoltenVK, the way Moonlight's Vulkan renderer does;
`numpy_forward.py` runs the network from the extracted weights.

Results on an M4 Pro (20-core GPU), macOS 27, `Images/artcnn_examples.png` luma (1600x1300):

| Implementation | PSNR vs ONNX (float32) | Max error (8-bit codes) |
|---|---|---|
| `translate.py --float32` | 138.6 dB | 1 (rounding) |
| NumPy from extracted weights | 137.6 dB | 1 (rounding) |
| `winograd.py` (half, shipped) | 59.6 dB | 2 |
| `translate.py` (half) | 59.5 dB | 2 |
| Original GLSL via libplacebo/MoltenVK | 56.0 dB | 3 |

Wall-clock time per frame, 300 pipelined frames:

| Implementation | 1280x800 | 1512x982 | 1920x1080 |
|---|---|---|---|
| Metal, Winograd on simdgroup_matrix (shipped) | 2.52 ms | 3.66 ms | 5.0-5.3 ms |
| Metal, direct translation | 4.51 ms | 6.23 ms | 9.00 ms |
| libplacebo/MoltenVK, original GLSL | 4.40 ms | 6.37 ms | 8.82 ms |

MPSGraph's float16 convolutions take 6.1-6.3 ms at 1512x982.

libplacebo's own per-pass GPU timers report about 0.5 ms at 1512x982 on MoltenVK, which
would exceed the GPU's peak arithmetic rate by 9x, so only wall-clock time is compared.

## How the 16->16 kernel works

Each threadgroup computes a 16x8 pixel block (32 Winograd tiles) with 128 threads:

1. Stage the input block plus a one-pixel border into threadgroup memory.
2. Each thread reads one tile for one group of 4 input channels into registers (B^T d).
3. In two batches of 8 transform positions, write the transformed tiles to threadgroup
   memory, multiply each position's (32 tiles x 16) block by its 16x16 weight matrix with
   8x8 `simdgroup_matrix` operations in place, and fold the products into the rows of the
   output transform.
4. Finish the output transform (A^T m A), add bias, apply ReLU, write 2x2 pixels.

One 8 KB threadgroup buffer serves both the staged input and the transformed tiles; at 22 KB
occupancy fell to 12% and the layer took 1.21 ms instead of 0.60 ms. Weight tiles are
filled per lane from a constant table using the lane-to-element layout of `simdgroup_matrix`,
measured on an M4 Pro: lane l holds row `(q & 4) + (l / 2) % 4`, columns
`(q & 2) * 2 + (l % 2) * 2` and the next one, with q = l / 4.

Per-layer counters before and after (1512x982):

| | Winograd, per-thread math | Winograd on simdgroup_matrix |
|---|---|---|
| Time per 16->16 layer | 0.98 ms | 0.60 ms |

A plain implicit-GEMM conv on `simdgroup_matrix` (no Winograd) also took 0.98 ms per layer,
with the F16 limiter at 94% and the occupancy manager no longer throttling: the matrix
hardware does 2.25x the multiplies of the per-thread Winograd kernel in the same time.

## Layout

The GLSL stores each 16-channel activation as a texture twice the input size, with the
four channel groups in a 2x2 block of texels. Here each one is a 4-slice texture array at
input size, so a SIMD group writing one channel group writes contiguous rows; the
`load`/`store` helpers map the GLSL's tiled coordinates onto slices. This cut last-level
cache traffic by 30% and frame time from 5.95 ms to 5.61 ms.

## Profiling

`tools/counters.py` summarizes GPU limiter counters from an `xctrace` recording made with an
Instruments template that has the Metal GPU Counters instrument set to "Performance
Limiters" (the counter set can only be chosen in the Instruments app; save it as a
template once, then record from the command line):

```
xcrun xctrace record --template ArtCNN --env ARTCNN_ONLY_PASS=1 --output layer.trace \
    --launch -- build/artcnn_metal_cli ArtCNN_C4F16_DS.metal in.f32 1512 982 out.f32 400
python tools/counters.py layer.trace
```

For register counts, spills and per-line costs, save one frame as a GPU trace and open it
in Xcode (Profile the trace, then look at the pipeline statistics of each `artcnn_pass`):

```
MTL_CAPTURE_ENABLED=1 ARTCNN_CAPTURE=frame.gputrace \
    build/artcnn_metal_cli ArtCNN_C4F16_DS.metal in.f32 1512 982 out.f32
```

These counters led to the current kernel: the earlier per-thread Winograd layer ran at 33%
occupancy (occupancy manager target 42%) with the F16 limiter at 73%, while the
`simdgroup_matrix` version is no longer throttled by the occupancy manager.

## Where the time goes

Per-pass profiling (`ARTCNN_PROFILE=1`) puts each 16->16 pass at about 0.60 ms. A kernel
that only copies one layer's activations (read and write 47 MB) takes 0.39 ms, close to the
memory bandwidth, so the layers are now within about 1.5x of the memory floor. Storing the
activations in 8 bits is the remaining way to lower that floor.

Things measured and rejected with the earlier per-thread Winograd kernel (5.54-5.95 ms per
frame), all with output matching it:

| Change | Result |
|---|---|
| Weights in a `constant` table instead of immediates | 23 ms |
| Output group per threadgroup (grid z) instead of per SIMD group | 2.0 ms per pass |
| Two output groups per thread | 1.4 ms per pass |
| Reading Winograd tiles straight from the texture | 1.22 ms per pass |
| Fusing two 16->16 layers, first layer recomputed on a 1-pixel border | 7.57 ms per frame |
| Not storing the first conv; recomputing it from luma in the passes that read it | 5.78 ms per frame |
| Two-half Winograd tiles to cut live registers | 6.29 ms per frame |

Folding depth-to-space into the skip-connection conv is kept: 5.61 ms to 5.54 ms, identical
output. Indexing the result vector in a loop there made the pass 5x slower; the generator
writes the four components explicitly. The same applies to arrays of `simdgroup_matrix`
tiles: indexed with loop variables, a layer took 11 ms instead of 0.98 ms.

Fusion removes a write and a read per pair but recomputes about 1.4x of the first layer
and leaves lanes idle, which costs more than the 0.39 ms it saves.
