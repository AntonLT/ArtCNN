# ArtCNN on Metal

Native Metal (MSL 4.1) versions of the ArtCNN GLSL shaders, generated from the official
`GLSL/*.glsl` files so the weights stay identical. Only `C4F16_DS` is generated so far.

- `translate.py` converts each mpv/libplacebo pass into a Metal compute kernel, line for line.
- `winograd.py` builds on it and replaces the five 16->16 convolutions with Winograd
  F(2x2, 3x3) kernels using weights extracted by `weights.py`. This is the shipped version.
  `--fuse a,b` runs two consecutive 16->16 layers in one kernel; it is currently slower.
  The skip-connection conv writes the final doubled luma directly, so there is no separate
  depth-to-space pass.
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

Wall-clock time per frame at 1512x982, 200-300 pipelined frames:

| Implementation | Time |
|---|---|
| Metal Winograd (shipped) | 5.54 ms |
| Metal direct translation | 6.23 ms |
| MPSGraph float16 convolutions | 6.10-6.26 ms |
| libplacebo/MoltenVK, original GLSL | 6.37 ms |

libplacebo's own per-pass GPU timers report about 0.5 ms at 1512x982 on MoltenVK, which
would exceed the GPU's peak arithmetic rate by 9x, so only wall-clock time is compared.

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

For a 16->16 Winograd layer the GPU holds occupancy at about 33% (occupancy manager
target 42%) with the F16 limiter at 73% and the last-level cache limiter at 85%; a plain
copy kernel reaches 89% occupancy. Raising occupancy is the next lever.

## Where the time goes

Per-pass profiling (`ARTCNN_PROFILE=1`) puts each 16->16 Winograd pass at about 1.05 ms.
A kernel that only copies one layer's activations (read and write 47 MB) takes 0.39 ms,
close to the memory bandwidth, so each pass is roughly half memory traffic and half
arithmetic that only partly overlaps it.

Things measured and rejected, all with output matching the shipped shader:

| Change | Result |
|---|---|
| Weights in a `constant` table instead of immediates | 23 ms |
| Output group per threadgroup (grid z) instead of per SIMD group | 2.0 ms per pass |
| Two output groups per thread | 1.4 ms per pass |
| Reading Winograd tiles straight from the texture | 1.22 ms per pass |
| Fusing two layers (`--fuse 1,2 3,4`), first layer recomputed on a 1-pixel border | 7.57 ms per frame |
| Not storing the first conv; recomputing it from luma in the passes that read it | 5.78 ms per frame |
| Two-half Winograd tiles to cut live registers | 6.29 ms per frame |

Folding depth-to-space into the skip-connection conv is kept: 5.61 ms to 5.54 ms, identical
output. Indexing the result vector in a loop there made the pass 5x slower; the generator
writes the four components explicitly.

Shipped shader by input size: 1280x800 3.83 ms, 1512x982 5.54 ms, 1920x1080 7.65 ms.

Fusion removes a write and a read per pair but recomputes about 1.4x of the first layer
and leaves lanes idle, which costs more than the 0.39 ms it saves. Larger fused tiles or
Winograd F(4x4, 3x3), which needs about 1.8x fewer multiplies but has larger transform
coefficients in half precision, are the remaining options.
