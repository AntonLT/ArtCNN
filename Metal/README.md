# ArtCNN on Metal

Native Metal (MSL 4.1) versions of the ArtCNN GLSL shaders, generated from the official
`GLSL/*.glsl` files so the weights stay identical. Only `C4F16_DS` is generated so far.

- `translate.py` converts each mpv/libplacebo pass into a Metal compute kernel, line for line.
- `winograd.py` builds on it and generates the shipped shader: the five 16->16 convolutions
  as Winograd F(4, 3) across x F(2, 3) down with the products on the 8x8 matrix hardware
  (`simdgroup_matrix`) and every tile kept in registers, using weights extracted by
  `weights.py`. The first conv is recomputed from luma where it is read, and the
  skip-connection conv writes the final doubled luma, so a frame takes six passes.
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
| Metal, shipped | 2.22 ms | 3.21 ms | 4.38 ms |
| Metal, direct translation | 4.51 ms | 6.23 ms | 9.00 ms |
| libplacebo/MoltenVK, original GLSL | 4.40 ms | 6.37 ms | 8.82 ms |

MPSGraph's float16 convolutions take 6.1-6.3 ms at 1512x982.

libplacebo's own per-pass GPU timers report about 0.5 ms at 1512x982 on MoltenVK, which
would exceed the GPU's peak arithmetic rate by 9x, so only wall-clock time is compared.

## Power

`tools/energy.py` renders frames paced at a fixed rate (waiting for each frame, like a stream)
and reads GPU and package power with `macmon` (IOReport, no sudo; `brew install macmon`),
averaging the steady part of each run. Two alternating runs per row, M4 Pro, 15 s each:

| Input | Rate | Metal GPU power | MoltenVK GPU power | Metal frames late | MoltenVK frames late |
|---|---|---|---|---|---|
| 1920x1080 | 24 fps | 3.4-3.8 W | 8.7-9.1 W | 0/360 | 5-6/360 |
| 1920x1080 | 60 fps | 7.5-9.3 W | 20.7-22.7 W | 1-2/900 | 25-27/900 |
| 1512x982 | 120 fps | 11.4-13.2 W | 32.5-33.4 W | 7-11/1800 | 117-134/1800 |

GPU energy above idle is about 2.5x lower with Metal at every rate (Full HD: 138-154 mJ
per frame against 343-377 mJ), more than the 1.7-1.8x difference in frame time. Whole-system
power at 60 fps Full HD is 19-22 W against 33-35 W. These are IOReport estimates from the
SoC's energy model; a paced loop in a real renderer may clock the GPU differently.

## First conv

The 1->16 first conv is not stored. The first 16->16 pass and the skip-connection pass stage
the luma block and compute its 16 channels per staged pixel (zero outside the image), which
removes a 47 MB write and two 47 MB reads: 3.51 ms per frame at 1512x982 instead of 3.66 ms,
with identical output. Computing it per (pixel, channel group) instead re-read the luma
neighborhood four times and gained nothing.

## Storage experiments

Measured after the first-conv change (3.51 ms per frame at 1512x982):

| Change | Result |
|---|---|
| Device buffers instead of texture arrays (copy kernel, one activation) | 0.41 ms vs 0.39 ms, no gain |
| 8-bit activations (RGBA8 storage, speed only) | 3.48 ms vs 3.52 ms, 1% |

8-bit storage would also cost quality: simulated in NumPy (`tools/quantize_sim.py`) with
per-channel scales from the maximum over a calibration set plus 10%, linear 8 bits gives
47-48 dB against the unquantized network on real images (up to 10 codes off) and square-root
companding 58 dB (up to 5 codes). With a 16->16 layer at 0.60 ms either way, the layers are
no longer limited by memory traffic: counters show the F16 limiter at 85% and instruction
throughput at 80%, mostly the 8x8 matrix products.

Winograd F(4x4, 3x3) would cut the matrix products by 1.8x, but simulated in float16
(`tools/winograd_precision.py`) it loses too much precision on fine detail: on a real image
crop 78.6 dB against float32 (max 1 code), on noise 50.2 dB with errors up to 24 codes,
where F(2x2, 3x3) stays at 86.2 dB and 64.7 dB with at most 1 code.

## How the 16->16 kernel works

Each threadgroup computes a 32x8 pixel block with 128 threads; SIMD group sg owns one row of
8 tiles of 4x2 pixels.

1. Stage the input block plus a one-pixel border (34x10 pixels) into threadgroup memory.
2. Lane l stands for row r = (q & 4) + (l / 2) % 4 and column pair c = (q & 2) * 2 + (l % 2) * 2
   of an 8x8 `simdgroup_matrix` (q = l / 4). It reads tile r's 6x4 window for channels c,
   c+1, 8+c and 9+c and computes the Winograd input transform in registers; the results are
   exactly its elements of the two input matrices for each of the 24 transform positions.
3. For each position, fill the four 8x8 weight tiles per lane from a constant table and
   multiply; the products land in the same lane, which folds them into the output transform.
4. Add bias, apply ReLU, exchange channel pairs with the neighbor lane (`simd_shuffle_xor`)
   and write full 4-channel texels.

Metal does not document the lane-to-element layout (measured on an M4 Pro; MLX relies on the
same one), so `ArtCNNMetal` checks it on the device at startup and fails initialization if it
differs. Loading weight tiles with `simdgroup_load` through threadgroup memory instead is
layout-independent but took 5.2 ms per frame where the per-lane fill took 3.66 ms.

Kernel history at 1512x982 (time per 16->16 layer, frame):

| Kernel | Layer | Frame | Notes |
|---|---|---|---|
| Winograd F(2x2), per-thread math | 0.98 ms | 5.54 ms | F16 limiter 73%, occupancy 33% |
| Direct conv on `simdgroup_matrix` | 0.98 ms | | 2.25x the multiplies, F16 limiter 94% |
| F(2x2) on `simdgroup_matrix`, 22 KB threadgroup memory | 1.21 ms | 6.64 ms | occupancy 12% |
| F(2x2) on `simdgroup_matrix`, 8 KB, batched through threadgroup memory | 0.60 ms | 3.66 ms | |
| F(2x2), tiles in registers | 0.59 ms | 3.43 ms | threadgroup traffic -4x, F16 limiter still 84% |
| F(4, 3) x F(2, 3), tiles in registers (shipped) | 0.55 ms | 3.21 ms | 25% fewer products |

With tiles in registers the layer is bound by half-precision arithmetic, and on M4 the
matrix operations run on those units, so the remaining lever is the number of products.
The hybrid tile has 24 transform positions per 8 output pixels instead of 16 per 4. Its
larger coefficients cost a little precision: against ONNX on the 1600x1300 test image, 44 of
8.3 million output values are off by 3-4 codes (1 before) and the mean error is 0.197 codes
(0.196 before). Full F(4x4, 3x3) would cut products further but fails the precision check below.

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
